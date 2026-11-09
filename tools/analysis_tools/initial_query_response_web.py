#!/usr/bin/env python
"""Validation browser for actual encoder class scores and sonar query priors.

Supports Grounding DINO, OAG and Sonarbind-Prism repositories. This tool adds
no fusion operation: every displayed additive term is captured from model code.
"""

import argparse
from collections import OrderedDict
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import sys
import threading
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw


def activate_repository(config, repo_root=None):
    """Choose model code before importing any MMDetection modules."""
    config = Path(config).expanduser().resolve()
    if repo_root is None:
        candidates = [
            parent for parent in config.parents
            if (parent / "mmdet/__init__.py").is_file()
            and (parent / "configs").is_dir()]
        if not candidates:
            raise ValueError("Cannot locate model repository; pass --repo-root.")
        root = candidates[0]
    else:
        root = Path(repo_root).expanduser().resolve()
    if not (root / "mmdet/__init__.py").is_file():
        raise ValueError("Repository root must contain mmdet/__init__.py.")
    loaded = sys.modules.get("mmdet")
    if loaded is not None:
        source = Path(loaded.__file__).resolve()
        if root not in source.parents:
            raise RuntimeError(
                "Another mmdet repository is already imported. Launch each "
                "repository in a separate Python process.")
    sys.path.insert(0, str(root))
    return root


def model_family(model):
    names = {cls.__name__ for cls in type(model).__mro__}
    if "OAG" in names:
        return "OAG"
    if "SonarBindPrism" in names:
        return "Sonarbind-Prism"
    if "GroundingDINO" in names:
        return "Grounding-DINO"
    raise ValueError("Unsupported detector: " + type(model).__name__)


class _CapturedReferences(Exception):
    pass


def capture_initial_queries(model, item):
    """Stop ordinary predict immediately before decoder attention."""
    import torch
    from mmengine.dataset import pseudo_collate

    captured = {}

    def capture(*args, **kwargs):
        refs = kwargs.get("reference_points")
        if refs is None or refs.ndim != 3 or refs.shape[0] != 1:
            raise RuntimeError("Expected single-image decoder reference points.")
        if refs.shape[-1] not in (2, 4):
            raise RuntimeError("Unsupported decoder reference format.")
        captured["references"] = refs[0].detach().float().cpu().numpy().copy()
        raise _CapturedReferences()

    model.eval()
    with torch.inference_mode():
        data = model.data_preprocessor(pseudo_collate([item]), training=False)
        meta = dict(data["data_samples"][0].metainfo)
        with patch.object(model.decoder, "forward", side_effect=capture):
            try:
                model(**data, mode="predict")
            except _CapturedReferences:
                pass
    if "references" not in captured:
        raise RuntimeError("Inference never reached the decoder.")
    return captured["references"], meta


def centers_in_original_image(references, meta):
    """Map padded-canvas references to the original RGB resize-only geometry."""
    if meta.get("flip", False):
        raise ValueError("Use a validation pipeline without flipping.")
    if "homography_matrix" in meta:
        matrix = np.asarray(meta["homography_matrix"])
        expected = np.eye(3)
        expected[0, 0], expected[1, 1] = matrix[0, 0], matrix[1, 1]
        if not np.allclose(matrix, expected):
            raise ValueError("Crop/rotation pipelines are unsupported.")
    ph, pw = meta["batch_input_shape"][:2]
    ih, iw = meta["img_shape"][:2]
    oh, ow = meta["ori_shape"][:2]
    scale = np.asarray(meta["scale_factor"], dtype=np.float64).reshape(-1)
    if scale.size not in (2, 4) or not np.isfinite(scale).all() or (scale[:2] <= 0).any():
        raise ValueError("Expected positive width/height scale factors.")
    resized = references[:, :2].astype(np.float64) * [pw, ph]
    pixels = resized / scale[:2]
    visible = (
        np.isfinite(pixels).all(-1)
        & (resized[:, 0] >= 0) & (resized[:, 0] < iw)
        & (resized[:, 1] >= 0) & (resized[:, 1] < ih)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < ow)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < oh))
    return pixels, visible


def class_responses(logits, mapping):
    """Match Grounding-DINO's mean sigmoid over each entity's text tokens."""
    import torch

    raw, activated, token_indices = [], [], []
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Explicit entity token spans are required.")
    if sorted(mapping) != list(range(1, len(mapping) + 1)):
        raise ValueError("Expected consecutive one-based entity token IDs.")
    for label in sorted(mapping):
        indices = sorted(set(int(t) for t in mapping[label]))
        if not indices or min(indices) < 0 or max(indices) >= logits.shape[-1]:
            raise ValueError("Entity tokens lie outside classification logits.")
        values = logits[:, indices]
        if not torch.isfinite(values).all():
            raise ValueError("An entity contains a masked/nonfinite text token.")
        raw.append(values.mean(-1))
        activated.append(values.sigmoid().mean(-1))
        token_indices.append(indices)
    return torch.stack(raw, -1).numpy(), torch.stack(
        activated, -1).numpy(), token_indices


def capture_responses(model, item, dataset_classes=()):
    """Capture full pre-top-k scores, then stop before decoder attention."""
    import torch
    captured = {}
    original_pre = model.pre_decoder
    original_proposals = model.gen_encoder_output_proposals

    def pre(*args, **kwargs):
        if args:
            raise ValueError("Expected named pre_decoder inputs.")
        captured["shapes"] = kwargs["spatial_shapes"].detach().cpu().numpy()
        mask = kwargs["memory_mask"]
        captured["padding"] = (
            None if mask is None else mask[0].detach().cpu().numpy())
        samples = kwargs["batch_data_samples"]
        captured["mapping"] = samples[0].get("token_positive_map")
        captured["text"] = samples[0].get("text", dataset_classes)
        return original_pre(*args, **kwargs)

    def proposals(*args, **kwargs):
        result = original_proposals(*args, **kwargs)
        captured["valid_proposals"] = torch.isfinite(
            result[1][0]).all(-1).detach().cpu().numpy()
        return result

    def semantic(module, inputs, output):
        captured["logits"] = output[0].detach().float().cpu()

    with ExitStack() as stack:
        stack.enter_context(patch.object(model, "pre_decoder", side_effect=pre))
        stack.enter_context(patch.object(
            model, "gen_encoder_output_proposals", side_effect=proposals))
        branch = model.bbox_head.cls_branches[model.decoder.num_layers]
        hook = branch.register_forward_hook(semantic)
        stack.callback(hook.remove)
        if hasattr(model, "rgb_prompt_mapper"):
            import mmdet.models.detectors.oag_rgb_prompt as prompt_module
            original_bias = prompt_module.prompt_token_bias

            def bias(*args, **kwargs):
                result = original_bias(*args, **kwargs)
                captured["bias"] = result[0].detach().float().cpu()
                captured["weight"] = float(model.sonar_position_weight)
                captured["addition_kind"] = "class_conditioned_rgb_prompt"
                return result

            stack.enter_context(patch.object(
                prompt_module, "prompt_token_bias", side_effect=bias))
        elif hasattr(model, "_position_scores"):
            original_position = model._position_scores

            def position(*args, **kwargs):
                result = original_position(*args, **kwargs)
                captured["bias"] = result[0].detach().float().cpu()[:, None]
                captured["weight"] = float(model.query_position_weight)
                captured["addition_kind"] = "class_agnostic_affine_prior"
                return result

            stack.enter_context(patch.object(
                model, "_position_scores", side_effect=position))
        refs, meta = capture_initial_queries(model, item)

    if "logits" not in captured:
        raise ValueError(
            "This configuration does not execute the text classification head.")
    if len(refs) != model.num_queries:
        raise ValueError("Unexpected initial query count.")
    pixels, visible = centers_in_original_image(refs, meta)
    logits = captured["logits"]
    raw, sigmoid, indices = class_responses(logits, captured["mapping"])
    arrays = dict(encoder_logit=raw, encoder_sigmoid=sigmoid)
    if "bias" in captured:
        bias = captured["bias"]
        if bias.shape[-1] == 1:
            bias = bias.expand_as(logits)
        if bias.shape != logits.shape:
            raise ValueError("Sonar bias and class logits have different shapes.")
        weight = captured["weight"]
        after, after_sigmoid, _ = class_responses(
            logits + weight * bias, captured["mapping"])
        contribution, _, _ = class_responses(
            weight * bias, captured["mapping"])
        np.testing.assert_allclose(
            after, raw + contribution, atol=2e-5, rtol=1e-5)
        arrays.update(
            selection_logit=after, selection_sigmoid=after_sigmoid,
            sonar_logit=contribution)
    count = raw.shape[1]
    prompt = captured["text"]
    if isinstance(prompt, (list, tuple)) and len(prompt) == count:
        names = [str(name) for name in prompt]
    elif len(dataset_classes) == count:
        names = [str(name) for name in dataset_classes]
    elif isinstance(prompt, str) and len(prompt.strip(". ").split(". ")) == count:
        names = prompt.strip(". ").split(". ")
    else:
        names = ["entity_{}".format(i + 1) for i in range(count)]
    valid = captured["valid_proposals"].copy()
    if captured["padding"] is not None:
        valid &= ~captured["padding"]
    if sum(int(h * w) for h, w in captured["shapes"]) != len(valid):
        raise ValueError("Encoder shapes do not match the proposal count.")
    if not valid.any():
        raise ValueError("No valid encoder proposals.")
    family = model_family(model)
    stages = ["encoder"]
    if "selection_logit" in arrays:
        stages.extend(["selection", "sonar"])
    note = {
        "Sonarbind-Prism":
            "RGB encoder proposal-head class response after multimodal fusion.",
        "OAG":
            "RGB-text encoder proposal-head response before sonar score addition.",
        "Grounding-DINO":
            "Grounding DINO RGB-text encoder proposal-head class response; "
            "this model has no sonar branch.",
    }[family]
    if family == "OAG" and "selection" not in stages:
        note += (
            " This model has no additive sonar ranking prior; no artificial "
            "post-addition field is generated.")
    return dict(
        arrays=arrays, shapes=captured["shapes"], valid=valid,
        references_normalized=refs, centers_original_xy=pixels, visible=visible,
        meta=meta, classes=names, token_indices=indices, stages=stages,
        family=family, model=type(model).__name__,
        addition_kind=captured.get("addition_kind"),
        weight=captured.get("weight", 0.0), note=note)


def rasterize(values, record, level="all"):
    """Interpolate valid pyramid grids on the padded canvas, then undo resize."""
    import torch
    import torch.nn.functional as F

    shapes, meta = record["shapes"], record["meta"]
    if level != "all":
        level = int(level)
        if not 0 <= level < len(shapes):
            raise ValueError("Feature level is outside the encoder pyramid.")
    ph, pw = map(int, meta["batch_input_shape"][:2])
    ih, iw = map(int, meta["img_shape"][:2])
    oh, ow = map(int, meta["ori_shape"][:2])
    if not (0 < ih <= ph and 0 < iw <= pw):
        raise ValueError("Invalid resized/padded image dimensions.")
    values = np.asarray(values)
    if values.shape != record["valid"].shape:
        raise ValueError("Expected one class score per RGB grid location.")
    numerator = torch.zeros(1, 1, ph, pw)
    denominator = torch.zeros_like(numerator)
    start = 0
    for index, (h, w) in enumerate(shapes):
        h, w = int(h), int(w)
        end = start + h * w
        if level == "all" or level == index:
            valid = record["valid"][start:end] & np.isfinite(values[start:end])
            grid = torch.from_numpy(np.where(
                valid, values[start:end], 0).copy()).reshape(1, 1, h, w)
            mask = torch.from_numpy(valid.astype(np.float32)).reshape(1, 1, h, w)
            numerator += F.interpolate(
                grid, (ph, pw), mode="bilinear", align_corners=False)
            denominator += F.interpolate(
                mask, (ph, pw), mode="bilinear", align_corners=False)
        start = end
    numerator = F.interpolate(
        numerator[:, :, :ih, :iw], (oh, ow),
        mode="bilinear", align_corners=False)
    denominator = F.interpolate(
        denominator[:, :, :ih, :iw], (oh, ow),
        mode="bilinear", align_corners=False)
    field = (numerator / denominator.clamp_min(1e-6))[0, 0].numpy()
    coverage = denominator[0, 0].numpy() > 1e-6
    return field, coverage


def select_values(record, stage, class_id, mode):
    if stage not in record["stages"]:
        raise ValueError("This model did not compute the requested stage.")
    if mode not in ("logit", "sigmoid"):
        raise ValueError("Unknown score mode.")
    # A sonar contribution is an additive score, not a sigmoid confidence.
    key = stage + "_" + ("logit" if stage == "sonar" else mode)
    values = record["arrays"][key]
    if class_id == "all":
        return values.max(-1)
    class_id = int(class_id)
    if not 0 <= class_id < values.shape[1]:
        raise ValueError("Class is outside the current prompt.")
    return values[:, class_id]


def overlay(image, field, coverage, lower, upper, alpha):
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Expected finite lower < upper.")
    if not 0 <= alpha <= 255:
        raise ValueError("Alpha must be in [0, 255].")
    if field.shape != (image.height, image.width):
        raise ValueError("Heatmap and original image dimensions differ.")
    t = np.clip((field - lower) / (upper - lower), 0, 1)
    rgb = np.stack([
        np.clip(1.5 - abs(4 * t - 3), 0, 1),
        np.clip(1.5 - abs(4 * t - 2), 0, 1),
        np.clip(1.5 - abs(4 * t - 1), 0, 1)], -1)
    mask = Image.fromarray(np.where(
        coverage & (field >= lower), alpha, 0).astype(np.uint8))
    return Image.composite(
        Image.fromarray((rgb * 255).astype(np.uint8)),
        image.convert("RGB"), mask)


def _query_marker(radius, supersample=6):
    """Render one anti-aliased marker: a saturated core ringed in white.

    Drawn at `supersample`x and downsampled with Lanczos so the circle stays
    smooth at any radius, matching the crisp dot markers used in paper
    figures rather than PIL's jagged native ellipses.
    """
    size = 2 * radius + 1
    big = size * supersample
    stamp = Image.new("RGBA", (big, big))
    draw = ImageDraw.Draw(stamp)
    c = big / 2
    outer = radius * supersample
    ring = max(1, round(supersample * 0.9))
    inner = max(0.0, outer - ring)
    draw.ellipse((c - outer, c - outer, c + outer, c + outer),
                 fill=(255, 255, 255, 235))
    draw.ellipse((c - inner, c - inner, c + inner, c + inner),
                 fill=(220, 35, 45, 235))
    return stamp.resize((size, size), Image.LANCZOS)


def query_overlay(image, record, radius=5):
    """Mark each visible initial-query center with a crisp, publication-style dot."""
    if radius <= 0:
        raise ValueError("Marker radius must be positive.")
    stamp = _query_marker(radius)
    half = stamp.width // 2
    layer = Image.new("RGBA", image.size)
    for x, y in record["centers_original_xy"][record["visible"]]:
        box = (int(round(x)) - half, int(round(y)) - half)
        layer.paste(stamp, box, stamp)
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


class ViewerState:
    def __init__(self, model, dataset, args):
        self.model, self.dataset, self.args = model, dataset, args
        self.lock, self.cache = threading.Lock(), OrderedDict()
        self.images = []
        for index in range(len(dataset)):
            info = dataset.get_data_info(index)
            self.images.append(dict(
                index=index, name=Path(info["img_path"]).name,
                img_id=str(info.get("img_id", ""))))

    def validate(self, index):
        if not 0 <= index < len(self.images):
            raise IndexError("Image index is outside the validation dataset.")

    def original(self, index):
        from mmengine.fileio import get

        self.validate(index)
        return Image.open(BytesIO(get(
            self.dataset.get_data_info(index)["img_path"]))).convert("RGB")

    def record(self, index):
        self.validate(index)
        with self.lock:
            if index not in self.cache:
                result = capture_responses(
                    self.model, self.dataset[index],
                    self.dataset.metainfo.get("classes", ()))
                if self.original(index).size != tuple(
                        reversed(result["meta"]["ori_shape"][:2])):
                    raise ValueError("Original image and metadata differ.")
                self.cache[index] = result
                if len(self.cache) > self.args.cache_size:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(index)
            return self.cache[index]

    def metadata(self, index, record):
        ranges = {}
        valid = record["valid"]
        for key, value in record["arrays"].items():
            ranges[key] = [
                [float(value[valid, c].min()), float(value[valid, c].max())]
                for c in range(value.shape[1])]
        return dict(
            index=index, name=self.images[index]["name"],
            classes=record["classes"], stages=record["stages"],
            shapes=record["shapes"].tolist(), ranges=ranges,
            family=record["family"], model=record["model"],
            addition_kind=record["addition_kind"],
            sonar_weight=record["weight"], note=record["note"],
            total_queries=len(record["visible"]),
            visible_queries=int(record["visible"].sum()),
            original_shape=list(record["meta"]["ori_shape"][:2]),
            score_definition="Entity mean of token logits, or mean of token "
            "sigmoids; full scores before top-k selection.",
            token_indices=record["token_indices"])


def create_server(state, host, port):
    html = Path(__file__).with_suffix(".html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def send(self, payload, content_type, status=200, filename=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header(
                    "Content-Disposition", 'attachment; filename="{}"'.format(
                        filename))
            self.end_headers()
            self.wfile.write(payload)

        def json(self, value, status=200):
            self.send(json.dumps(
                value, ensure_ascii=False, allow_nan=False).encode(),
                "application/json; charset=utf-8", status)

        def do_GET(self):
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                if path == "/":
                    return self.send(html, "text/html; charset=utf-8")
                if path == "/api/info":
                    return self.json(dict(
                        total=len(state.images), split="val",
                        config=state.args.config,
                        checkpoint=state.args.checkpoint,
                        repo_root=getattr(state.args, "repo_root", None),
                        model=type(state.model).__name__,
                        device=state.args.device, alpha=state.args.alpha))
                if path == "/api/images":
                    term = q.get("q", "").casefold()
                    offset = max(0, int(q.get("offset", 0)))
                    found = [v for v in state.images if
                             term in v["name"].casefold() or
                             term in (str(v["index"]), v["img_id"].casefold())]
                    return self.json(dict(
                        total=len(found), images=found[offset:offset + 40]))
                allowed = (
                    "/api/original/", "/api/queries/", "/api/heatmap/",
                    "/api/points/", "/api/download/")
                if not path.startswith(allowed):
                    return self.json(dict(error="Not found"), 404)
                index = int(path.rsplit("/", 1)[1])
                state.validate(index)
                if path.startswith("/api/original/"):
                    buffer = BytesIO()
                    state.original(index).save(buffer, format="PNG")
                    return self.send(buffer.getvalue(), "image/png")
                record = state.record(index)
                if path.startswith("/api/queries/"):
                    return self.json(state.metadata(index, record))
                buffer = BytesIO()
                if path.startswith("/api/download/"):
                    arrays = dict(record["arrays"])
                    arrays.update(
                        spatial_shapes=record["shapes"], valid=record["valid"],
                        references_normalized=record["references_normalized"],
                        centers_original_xy=record["centers_original_xy"],
                        visible=record["visible"],
                        metadata_json=np.array(json.dumps(
                            state.metadata(index, record), ensure_ascii=False)))
                    np.savez_compressed(buffer, **arrays)
                    return self.send(
                        buffer.getvalue(), "application/octet-stream",
                        filename="val_{}_encoder_responses.npz".format(index))
                image = state.original(index)
                if path.startswith("/api/points/"):
                    image = query_overlay(image, record)
                else:
                    stage = q.get("stage", "encoder")
                    values = select_values(
                        record, stage, q.get("class", "all"),
                        q.get("mode", "logit"))
                    field, coverage = rasterize(
                        values, record, q.get("level", "all"))
                    low = float(q.get("lower", float(values[record["valid"]].min())))
                    high = float(q.get("upper", float(values[record["valid"]].max())))
                    if high == low:
                        high = low + 1e-6
                    image = overlay(
                        image, field, coverage, low, high,
                        int(q.get("alpha", state.args.alpha)))
                    if q.get("points", "0") == "1":
                        image = query_overlay(image, record)
                image.save(buffer, format="PNG")
                return self.send(
                    buffer.getvalue(), "image/png",
                    filename="val_{}_{}.png".format(
                        index, q.get("stage", "queries")))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, IndexError) as error:
                self.json(dict(error=str(error)), 400)
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.json(dict(error="{}: {}".format(
                    type(error).__name__, error)), 500)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def load_model_dataset(args):
    root = activate_repository(args.config, getattr(args, "repo_root", None))
    args.repo_root = str(root)
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import DATASETS, MODELS
    from mmdet.utils import register_all_modules

    torch.set_num_threads(4)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.model.get("test_cfg", {}).get("chunked_size", -1) > 0:
        raise ValueError("Set model.test_cfg.chunked_size=-1.")
    register_all_modules(init_default_scope=False)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=True)
    model.to(args.device).eval()
    model_family(model)
    if getattr(model, "use_text_branch", True) is False or getattr(
            model, "language_model", None) is None:
        raise ValueError("Class response maps require the model's text branch.")
    return model, dataset


def main():
    from mmengine.config import DictAction

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--repo-root", help="Model repository; inferred from config by default.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--alpha", type=int, default=180)
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()
    if not (1 <= args.port <= 65535 and 0 <= args.alpha <= 255
            and 1 <= args.cache_size <= 32):
        parser.error("Invalid port, alpha or cache-size.")
    model, dataset = load_model_dataset(args)
    state = ViewerState(model, dataset, args)
    server = create_server(state, args.host, args.port)
    print("Encoder class response viewer: http://{}:{} | {} | {} validation "
          "images".format(args.host, server.server_port,
                          type(model).__name__, len(dataset)), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
