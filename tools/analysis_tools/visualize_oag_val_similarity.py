#!/usr/bin/env python3
"""Export validation-only OAG cross-modal cosine similarity heatmaps.

Run from the MMDetection repository root. This script never trains the model.
Text features are RGB-conditioned encoder outputs, not independent captions.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import types
from pathlib import Path

import numpy as np


def natural_key(value):
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", str(value))]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def similarity_statistics(matrix):
    diagonal = np.diag(matrix)
    n = len(diagonal)
    off_mean = ((matrix.sum(dtype=np.float64) - diagonal.sum(dtype=np.float64))
                / (n * (n - 1))) if n > 1 else None
    rows = np.arange(n)
    # Count ties as failures, rather than favoring the first sample in a tie.
    strict_row = np.sum(matrix >= diagonal[:, None], axis=1) == 1
    strict_col = np.sum(matrix >= diagonal[None, :], axis=0) == 1
    return dict(
        samples=n,
        min_cosine=float(matrix.min()),
        max_cosine=float(matrix.max()),
        mean_paired_cosine=float(diagonal.mean()),
        median_paired_cosine=float(np.median(diagonal)),
        mean_unpaired_cosine=float(off_mean) if off_mean is not None else None,
        mean_paired_minus_unpaired=(float(diagonal.mean() - off_mean)
                                   if off_mean is not None else None),
        row_retrieval_top1=float(np.mean(matrix.argmax(axis=1) == rows)),
        column_retrieval_top1=float(np.mean(matrix.argmax(axis=0) == rows)),
        row_retrieval_top1_strict=float(strict_row.mean()),
        column_retrieval_top1_strict=float(strict_col.mean()),
        retrieval_note="Exact-frame retrieval; same-class/nearby frames are "
                       "treated as nonmatches. This is not detection accuracy.",
    )


def draw_heatmap(matrix, title, ylabel, xlabel, destination, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(8.4, 7.4))
        fig.subplots_adjust(left=0.15, right=0.86, bottom=0.12, top=0.88)
        image = ax.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0,
                          origin="upper", interpolation="nearest",
                          aspect="equal", rasterized=True)
        n = matrix.shape[0]
        ticks = np.unique(np.linspace(0, n - 1, min(6, n)).astype(int))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(i + 1) for i in ticks])
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(i + 1) for i in ticks])
        ax.set_xlabel(xlabel + " sample index")
        ax.set_ylabel(ylabel + " sample index")
        ax.set_title(title + "\nValidation set, N={}; shared filename order".format(
            n), fontsize=11, pad=12)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                     label="Cosine similarity")
        fig.savefig(str(destination.with_suffix(".png")), dpi=dpi,
                    bbox_inches="tight", pad_inches=0.15)
        fig.savefig(str(destination.with_suffix(".pdf")), dpi=dpi,
                    bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)


def export_outputs(output, features, samples, metadata, dpi):
    output.mkdir(parents=True, exist_ok=True)
    for name, values in features.items():
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Invalid exported features: " + name)
        if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-4):
            raise ValueError("Exported features are not unit vectors: " + name)
        if len(values) != len(samples):
            raise ValueError("Feature/sample count mismatch: " + name)

    sr = np.clip(features["sonar"] @ features["rgb"].T, -1.0, 1.0)
    st = np.clip(features["sonar"] @ features["text_rgb_conditioned"].T,
                 -1.0, 1.0)
    np.savez_compressed(str(output / "features.npz"), **features)
    np.savez_compressed(str(output / "similarity_matrices.npz"),
                        sonar_rgb=sr, sonar_text_rgb_conditioned=st)

    with (output / "samples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    with (output / "paired_similarities.csv").open(
            "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index_1based", "img_id", "filename",
                         "sonar_rgb_cosine", "sonar_text_conditioned_cosine"])
        for i, sample in enumerate(samples):
            writer.writerow([i + 1, sample["img_id"], sample["filename"],
                             float(sr[i, i]), float(st[i, i])])

    draw_heatmap(sr, "Sonar-RGB feature similarity", "Sonar", "RGB",
                 output / "sonar_rgb_similarity", dpi)
    draw_heatmap(st, "Sonar / RGB-conditioned text feature similarity",
                 "Sonar", "RGB-conditioned text",
                 output / "sonar_text_conditioned_similarity", dpi)

    metadata["plot_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    metadata["statistics"] = {
        "sonar_rgb": similarity_statistics(sr),
        "sonar_text_rgb_conditioned": similarity_statistics(st),
    }
    metadata["interpretation"] = [
        "Diagonal cells are matching frame IDs; no diagonal boost is applied.",
        "Distance to the diagonal is index distance, not a semantic metric.",
        "Both axes use the same natural filename order; no similarity sorting.",
        "Text is an encoder output after RGB-text interaction. Input category "
        "prompts are shared across frames, not unique frame descriptions.",
        "Both figures use raw cosine similarity and the fixed [-1, 1] scale; "
        "no softmax, per-row rescaling, or smoothing is applied.",
        "Dataset-wide matrices extend the training loss's local batch "
        "similarities; they are not themselves the training objective.",
    ]
    write_json(output / "summary.json", metadata)
    return metadata


def extract_validation(args):
    import torch
    import torch.nn.functional as F
    from mmengine.config import Config
    from mmengine.runner import Runner, set_random_seed
    from mmdet.apis import init_detector

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg = Config.fromfile(str(config_path))
    cfg.launcher = "none"
    set_random_seed(args.seed, deterministic=False)
    torch.backends.cudnn.benchmark = False
    model = init_detector(cfg, str(checkpoint_path), device=args.device)
    model.eval()
    if not getattr(model, "enable_legacy_contrastive", False):
        print("NOTE: legacy contrastive losses are disabled in this config.",
              flush=True)

    loader_cfg = cfg.val_dataloader.copy()
    loader_cfg.batch_size = args.batch_size
    loader_cfg.num_workers = args.workers
    loader_cfg.persistent_workers = args.workers > 0
    loader_cfg.sampler = dict(type="DefaultSampler", shuffle=False)
    loader_cfg.pop("batch_sampler", None)
    loader = Runner.build_dataloader(loader_cfg, seed=args.seed)
    dataset = loader.dataset
    original_count = len(dataset)
    entries = [dataset.get_data_info(i) for i in range(original_count)]
    order = sorted(range(original_count),
                   key=lambda i: natural_key(entries[i]["img_path"]))
    if args.limit:
        order = order[:args.limit]
    if not order:
        raise ValueError("Validation dataset is empty.")

    loader_cfg.dataset = torch.utils.data.Subset(dataset, order)
    loader = Runner.build_dataloader(loader_cfg, seed=args.seed)
    blocks = {"sonar": [], "rgb": [], "text_rgb_conditioned": []}
    samples = []
    original = model.pre_decoder
    captured = []
    token_shapes = []

    def capture(self, *positional, **keywords):
        result = original(*positional, **keywords)
        head = result[1]
        pool = self.contrastive_loss_fn._masked_pool
        pooled = {}
        for name, key, mask_key, invert in [
                ("sonar", "memory_sonar", "memory_sonar_mask", True),
                ("rgb", "memory", "memory_mask", True),
                ("text_rgb_conditioned", "memory_text", "text_token_mask",
                 False)]:
            value = head[key].float()
            mask = head.get(mask_key)
            valid = ~mask if mask is not None and invert else mask
            pooled[name] = F.normalize(pool(value, valid), dim=-1).cpu().numpy()
        captured.append(pooled)
        if not token_shapes:
            token_shapes.append({k: list(head[k].shape) for k in
                                 ("memory_sonar", "memory", "memory_text")})
        return result

    model.pre_decoder = types.MethodType(capture, model)
    start = time.monotonic()
    print("VALIDATION_START samples={} batch_size={} device={}".format(
        len(order), args.batch_size, args.device), flush=True)
    try:
        with torch.inference_mode():
            for step, batch in enumerate(loader, 1):
                captured.clear()
                model.test_step(batch)
                if len(captured) != 1:
                    raise RuntimeError("Expected one pre_decoder call per "
                                       "batch; chunked prompts are unsupported.")
                for name in blocks:
                    blocks[name].append(captured[0][name])
                for sample in batch["data_samples"]:
                    info = sample.metainfo
                    text = sample.get("text", None)
                    if isinstance(text, (list, tuple)):
                        text = json.dumps(list(text), ensure_ascii=False)
                    samples.append(dict(
                        index_1based=len(samples) + 1,
                        img_id=int(info["img_id"]),
                        filename=Path(info["img_path"]).name,
                        rgb_path=info["img_path"],
                        sonar_path=info.get("sonar_path", ""),
                        input_text=text,
                    ))
                if step == 1 or step % args.log_interval == 0:
                    elapsed = time.monotonic() - start
                    print("PROGRESS {}/{} elapsed={:.1f}s rate={:.2f}/s".format(
                        len(samples), len(order), elapsed,
                        len(samples) / max(elapsed, 1e-9)), flush=True)
    finally:
        model.pre_decoder = original
    if len(samples) != len(order):
        raise RuntimeError("Incomplete validation extraction.")
    actual_ids = [s["img_id"] for s in samples]
    expected_ids = [int(entries[i]["img_id"]) for i in order]
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError("Sample IDs do not preserve the unique paired order.")

    features = {k: np.concatenate(v, axis=0).astype(np.float32)
                for k, v in blocks.items()}
    metadata = dict(
        config=str(config_path), checkpoint=str(checkpoint_path),
        checkpoint_bytes=checkpoint_path.stat().st_size,
        checkpoint_mtime=checkpoint_path.stat().st_mtime,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset_split="validation", full_validation_count=original_count,
        exported_count=len(samples), is_full_validation=len(samples) ==
        original_count, seed=args.seed, device=args.device,
        batch_size=args.batch_size,
        ordering="Natural ascending RGB image path; identical rows/columns",
        pooling="Same masked mean and L2 normalization as ContrastiveLoss; "
        "features captured from pre_decoder head output.",
        first_batch_token_shapes=token_shapes[0],
        loss_sonar_image_weight=model.loss_sonar_image_weight,
        loss_sonar_text_weight=model.loss_sonar_text_weight,
        text_rgb_conditioned=True,
        extraction_seconds=time.monotonic() - start,
    )
    cfg.dump(str(Path(args.out_dir) / "resolved_config.py"))
    return features, samples, metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?")
    parser.add_argument("checkpoint", nargs="?")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Use 1 to avoid other images changing padding.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0,
                        help="Smoke test only; 0 exports all validation pairs.")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--replot", metavar="EXISTING_OUTPUT_DIR",
                        help="Plot exported features without loading the model.")
    args = parser.parse_args()
    if min(args.batch_size, args.log_interval, args.dpi) < 1:
        parser.error("batch-size, log-interval and dpi must be positive")
    if min(args.workers, args.limit) < 0:
        parser.error("workers and limit cannot be negative")
    if not args.replot and (not args.config or not args.checkpoint):
        parser.error("config and checkpoint are required unless using --replot")
    return args


def main():
    args = parse_args()
    output = Path(args.out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    if args.replot:
        source = Path(args.replot)
        with np.load(str(source / "features.npz")) as data:
            features = {name: data[name] for name in data.files}
        with (source / "samples.csv").open(encoding="utf-8", newline="") as f:
            samples = list(csv.DictReader(f))
        metadata = json.loads((source / "summary.json").read_text(
            encoding="utf-8"))
        metadata["replotted_from"] = str(source.resolve())
    else:
        features, samples, metadata = extract_validation(args)
    metadata = export_outputs(output, features, samples, metadata, args.dpi)
    print("STATISTICS " + json.dumps(metadata["statistics"]), flush=True)
    print("COMPLETE " + str(output), flush=True)


if __name__ == "__main__":
    # Also works as tools/analysis_tools/<script>.py without pip-installing mmdet.
    repository = Path(__file__).resolve().parents[2]
    if (repository / "mmdet").is_dir():
        sys.path.insert(0, str(repository))
    main()
