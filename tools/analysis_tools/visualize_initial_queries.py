#!/usr/bin/env python
"""Overlay actual pre-decoder reference centers on original RGB images.

Run from the repository root, for example:
python tools/analysis_tools/visualize_initial_queries.py CONFIG CHECKPOINT \
    --device cuda:0 --num-images 5 --output-dir work_dirs/query_visualization
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

# Allow direct invocation without relying on an editable mmdet installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import pseudo_collate
from mmengine.fileio import get
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings
from PIL import Image, ImageDraw
from io import BytesIO

from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules


class _CapturedReferences(Exception):
    """Stop inference once decoder inputs are captured (no decoding needed)."""


def capture_initial_queries(model, item):
    """Return actual decoder references and post-preprocessor image metadata."""
    captured = {}

    def capture(*args, **kwargs):
        references = kwargs.get('reference_points')
        if references is None:
            raise RuntimeError('Decoder did not receive reference_points as a keyword.')
        if references.ndim != 3 or references.shape[0] != 1:
            raise RuntimeError('Expected one image and [1, queries, 2 or 4] references.')
        if references.shape[-1] not in (2, 4):
            raise RuntimeError('Unsupported reference point format.')
        captured['references'] = references[0].detach().float().cpu().numpy().copy()
        raise _CapturedReferences()

    model.eval()
    with torch.inference_mode():
        data = model.data_preprocessor(pseudo_collate([item]), training=False)
        captured['meta'] = dict(data['data_samples'][0].metainfo)
        # Stop immediately before decoder attention, not at final predictions.
        # Evaluation mode means DN training queries are never appended.
        with patch.object(model.decoder, 'forward', side_effect=capture):
            try:
                model(**data, mode='predict')
            except _CapturedReferences:
                pass
    if 'references' not in captured:
        raise RuntimeError('Inference never reached the decoder.')
    return captured['references'], captured['meta']


def capture_prior_heatmaps(model, item):
    """Capture full, pre-top-k semantic and affine sonar ranking fields.

    Pyramid values are rasterized on their encoder grids, averaged across
    valid levels, cropped to the resized image and resized to original RGB.
    Raw semantic logits and unweighted sonar responses are retained.
    """
    import inspect
    from contextlib import ExitStack
    import torch.nn.functional as F

    is_prompt = hasattr(model, 'rgb_prompt_mapper')
    if not is_prompt and not hasattr(model, '_position_scores'):
        raise ValueError('双先验热力图需要 OAGRGBPrompt / OAGStaged 或 OAGAffine 系列模型；'
                         '当前模型没有受支持的 RGB 声呐位置先验。')
    captured = {}
    original_pre = model.pre_decoder
    original_position = getattr(model, '_position_scores', None)

    def pre(*args, **kwargs):
        # Derived wrappers can expose only *args/**kwargs.
        from mmdet.models.detectors.oag_affine import OAGAffine
        from mmdet.models.detectors.oag_rgb_prompt import OAGRGBPrompt
        implementation = OAGRGBPrompt if is_prompt else OAGAffine
        bound = inspect.signature(implementation.pre_decoder).bind(model, *args, **kwargs)
        captured['shapes'] = bound.arguments['spatial_shapes'].detach().cpu().tolist()
        mask = bound.arguments['memory_mask']
        captured['mask'] = None if mask is None else mask[0].detach().cpu()
        return original_pre(*args, **kwargs)

    def semantic(module, inputs, output):
        captured['semantic'] = output[0].max(-1).values.detach().float().cpu()

    def position(*args, **kwargs):
        result = original_position(*args, **kwargs)
        captured['sonar'] = result[0].detach().float().cpu()
        return result

    def token_bias(*args, **kwargs):
        result = original_bias(*args, **kwargs)
        captured['sonar'] = result[0].max(-1).values.detach().float().cpu()
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(model, 'pre_decoder', side_effect=pre))
        if is_prompt:
            import mmdet.models.detectors.oag_rgb_prompt as prompt_module
            original_bias = prompt_module.prompt_token_bias
            stack.enter_context(patch.object(prompt_module, 'prompt_token_bias', side_effect=token_bias))
        else:
            stack.enter_context(patch.object(model, '_position_scores', side_effect=position))
        hook = model.bbox_head.cls_branches[model.decoder.num_layers].register_forward_hook(semantic)
        stack.callback(hook.remove)
        _, meta = capture_initial_queries(model, item)
    if not all(k in captured for k in ('semantic', 'sonar')):
        raise ValueError('当前配置没有执行语义和声呐位置联合先验选择。')
    # Reuse geometry validation (reject crop/rotation/flip).
    centers_in_original_image(np.zeros((1, 2)), meta)
    ph, pw = meta['batch_input_shape'][:2]
    ih, iw = meta['img_shape'][:2]
    oh, ow = meta['ori_shape'][:2]
    fields = {}
    for name in ('semantic', 'sonar'):
        values = captured[name]
        valid = torch.isfinite(values)
        if captured['mask'] is not None:
            valid &= ~captured['mask']
        numerator = torch.zeros((1, 1, ph, pw))
        denominator = torch.zeros_like(numerator)
        start = 0
        for h, w in captured['shapes']:
            end = start + h*w
            v = values[start:end].masked_fill(~valid[start:end], 0).reshape(1, 1, h, w)
            weight = valid[start:end].float().reshape(1, 1, h, w)
            numerator += F.interpolate(v, (ph, pw), mode='bilinear', align_corners=False)
            denominator += F.interpolate(weight, (ph, pw), mode='bilinear', align_corners=False)
            start = end
        if start != len(values):
            raise ValueError('Prior length does not match encoder pyramid.')
        grid = (numerator / denominator.clamp_min(1e-6))[:, :, :ih, :iw]
        fields[name] = F.interpolate(grid, (oh, ow), mode='bilinear', align_corners=False)[0, 0].numpy()
    return fields, meta


def prior_overlay(image, field, alpha, threshold=0.):
    """Blue-to-red colormap, independently min/max scaled per image/prior."""
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be in [0, 1]')
    low, high = float(field.min()), float(field.max())
    t = (field-low)/(high-low) if high > low else np.zeros_like(field)
    rgb = np.stack([np.clip(1.5-np.abs(4*t-3), 0, 1),
                    np.clip(1.5-np.abs(4*t-2), 0, 1),
                    np.clip(1.5-np.abs(4*t-1), 0, 1)], -1)
    color = Image.fromarray((rgb*255).astype(np.uint8))
    mask = Image.fromarray(np.where(t >= threshold, alpha, 0).astype(np.uint8))
    return Image.composite(color, image.convert('RGB'), mask)


def centers_in_original_image(references, meta):
    """Map normalized padded-canvas centers back to unresized RGB pixels."""
    if meta.get('flip', False):
        raise ValueError('Flipped pipelines are unsupported; use the standard val/test pipeline.')
    if 'homography_matrix' in meta:
        # The standard pipelines only resize; fail on unexpected crop/rotation.
        matrix = np.asarray(meta['homography_matrix'])
        expected = np.eye(3)
        expected[0, 0], expected[1, 1] = matrix[0, 0], matrix[1, 1]
        if not np.allclose(matrix, expected):
            raise ValueError('Crop/rotation transforms need inverse homography; use val/test resize only.')
    padded_h, padded_w = meta['batch_input_shape'][:2]
    image_h, image_w = meta['img_shape'][:2]
    original_h, original_w = meta['ori_shape'][:2]
    scale = np.asarray(meta['scale_factor'], dtype=np.float64).reshape(-1)
    if scale.size not in (2, 4) or (scale[:2] <= 0).any():
        raise ValueError('Expected a positive (width_scale, height_scale) scale_factor.')
    resized = references[:, :2].astype(np.float64) * [padded_w, padded_h]
    pixels = resized / scale[:2]
    visible = (np.isfinite(pixels).all(axis=1)
               & (resized[:, 0] >= 0) & (resized[:, 0] < image_w)
               & (resized[:, 1] >= 0) & (resized[:, 1] < image_h)
               & (pixels[:, 0] >= 0) & (pixels[:, 0] < original_w)
               & (pixels[:, 1] >= 0) & (pixels[:, 1] < original_h))
    # Do not clamp padded centers to the image border: that fabricates clusters.
    return pixels, visible


def save_overlay(image, pixels, visible, png_path, max_points, seed, radius, alpha):
    indices = np.flatnonzero(visible)
    if max_points and len(indices) > max_points:
        indices = np.sort(np.random.default_rng(seed).choice(indices, max_points, replace=False))
    base = image.convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for index in indices:
        x, y = pixels[index]
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(255, 40, 40, alpha))
    Image.alpha_composite(base, overlay).convert('RGB').save(png_path)
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--split', choices=['val'], default='val', help='Validation dataset only.')
    parser.add_argument('--web', action='store_true', help='Open the interactive validation-image browser.')
    parser.add_argument('--host', default='127.0.0.1', help='Web bind address; use 0.0.0.0 for LAN access.')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--output-dir', default='work_dirs/initial_query_visualization')
    parser.add_argument('--num-images', type=int, default=5)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--indices', nargs='+', type=int, help='Explicit zero-based dataset indices; overrides start/num-images.')
    parser.add_argument('--max-points', type=int, default=0, help='0 draws all queries; otherwise uniform random subset, NOT score top-k.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducible point subsampling.')
    parser.add_argument('--point-radius', type=float, default=2.)
    parser.add_argument('--alpha', type=int, default=180, help='Point opacity, 1 to 255.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be in [1, 65535].')
    if args.num_images <= 0 or args.start_index < 0 or args.max_points < 0:
        parser.error('num-images must be positive; start-index and max-points must be nonnegative.')
    if args.point_radius <= 0 or not 1 <= args.alpha <= 255:
        parser.error('point-radius must be positive; alpha must be in [1, 255].')
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.model.get('test_cfg', {}).get('chunked_size', -1) > 0:
        raise ValueError('Chunked text prompts are unsupported; set '
                         'model.test_cfg.chunked_size=-1 to use one prompt.')
    register_all_modules(init_default_scope=False)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    loader = cfg.get(args.split + '_dataloader')
    if not loader or not loader.get('dataset'):
        raise ValueError(f'Config has no {args.split} dataset.')
    dataset = DATASETS.build(loader.dataset)
    if args.web:
        indices = []
    else:
        indices = args.indices if args.indices is not None else list(
            range(args.start_index, min(args.start_index + args.num_images, len(dataset))))
    if not args.web and (not indices or any(i < 0 or i >= len(dataset) for i in indices)):
        raise ValueError(f'Dataset has {len(dataset)} images; requested indices: {indices}')
    # No init_weights(): all model tensors must come from the supplied checkpoint.
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    model.to(args.device).eval()
    if args.web:
        from initial_query_web import serve_viewer
        serve_viewer(model, dataset, args, capture_initial_queries,
                     centers_in_original_image, capture_prior_heatmaps, prior_overlay)
        return
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index in indices:
        item = dataset[index]
        references, meta = capture_initial_queries(model, item)
        if references.shape[0] != model.num_queries:
            raise RuntimeError('Expected exactly num_queries matching queries in eval mode. '
                               'Chunked prompts/TTA are not supported.')
        pixels, visible = centers_in_original_image(references, meta)
        image = Image.open(BytesIO(get(meta['img_path']))).convert('RGB')
        if image.size != tuple(reversed(meta['ori_shape'][:2])):
            raise ValueError('Original image dimensions differ from pipeline metadata.')
        stem = f'{index:06d}_{Path(meta["img_path"]).stem}'
        png = output / (stem + '_queries.png')
        shown = save_overlay(image, pixels, visible, png, args.max_points,
                             args.seed, args.point_radius, args.alpha)
        # Keep ALL references for comparisons, even when only a subset is drawn.
        npz = output / (stem + '_queries.npz')
        np.savez_compressed(npz, references_normalized=references,
                            centers_original_xy=pixels, visible=visible,
                            drawn_query_indices=shown)
        record = dict(dataset_index=index, img_id=str(meta.get('img_id', '')),
                      img_path=meta['img_path'], sonar_path=meta.get('sonar_path'),
                      total_queries=len(references), visible_queries=int(visible.sum()),
                      drawn_queries=len(shown), image=str(png), data=str(npz),
                      original_shape=list(meta['ori_shape'][:2]),
                      resized_shape=list(meta['img_shape'][:2]),
                      padded_shape=list(meta['batch_input_shape'][:2]),
                      scale_factor=np.asarray(meta['scale_factor']).tolist())
        records.append(record)
        print(f'[{index}] {len(references)} queries, {len(shown)} drawn -> {png}', flush=True)
    (output / 'manifest.json').write_text(json.dumps(
        dict(config=str(Path(args.config).resolve()), checkpoint=args.checkpoint,
             split=args.split, max_points=args.max_points, seed=args.seed,
             point_radius=args.point_radius, alpha=args.alpha, samples=records),
        indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
