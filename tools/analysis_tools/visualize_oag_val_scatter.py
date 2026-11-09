"""Plot paired validation features in one shared t-SNE embedding."""

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        raise ValueError("Choose an empty output directory.")

    keys = ("sonar", "rgb", "text_rgb_conditioned")
    feature_path = args.source / "features.npz"
    with np.load(feature_path, allow_pickle=False) as archive:
        arrays = [archive[key].astype(np.float32) for key in keys]
    n = arrays[0].shape[0]
    if n < 31 or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("Expected at least 31 aligned samples per modality.")
    for array in arrays:
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite input features.")
        np.testing.assert_allclose(
            np.linalg.norm(array, axis=1), 1.0, atol=1e-5)
    with (args.source / "samples.csv").open(
            encoding="utf-8-sig", newline="") as stream:
        samples = list(csv.DictReader(stream))
    if len(samples) != n:
        raise ValueError("Feature and sample counts differ.")

    parameters = dict(
        n_components=2, perplexity=30.0, early_exaggeration=12.0,
        learning_rate="auto", metric="cosine", init="pca",
        random_state=args.seed, method="barnes_hut", angle=0.5, n_jobs=4)
    iteration_key = (
        "max_iter" if "max_iter" in inspect.signature(TSNE).parameters
        else "n_iter")
    parameters[iteration_key] = 1500
    print("Embedding {} vectors jointly ...".format(3 * n), flush=True)
    with threadpool_limits(limits=4):
        reducer = TSNE(**parameters)
        embedded = reducer.fit_transform(np.concatenate(arrays, axis=0))
    if embedded.shape != (3 * n, 2) or not np.isfinite(embedded).all():
        raise ValueError("Invalid embedding.")
    coordinates = dict(zip(keys, np.split(embedded, 3)))
    selected = np.sort(np.random.default_rng(args.seed).choice(
        n, size=min(24, n), replace=False))
    np.savez_compressed(
        args.out_dir / "scatter_coordinates.npz", **coordinates,
        linked_sample_indices_zero_based=selected)

    low, high = embedded.min(axis=0), embedded.max(axis=0)
    margin = np.maximum((high - low) * 0.06, 1.0)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    for key, label, color, marker, filename in (
        ("rgb", "RGB", "#D85448", "^", "sonar_rgb_scatter"),
        ("text_rgb_conditioned", "Text (RGB-conditioned)", "#A97509", "D",
         "sonar_text_conditioned_scatter"),
    ):
        fig, ax = plt.subplots(figsize=(8.4, 6.8), constrained_layout=True)
        sonar, other = coordinates["sonar"], coordinates[key]
        segments = np.stack((sonar[selected], other[selected]), axis=1)
        ax.add_collection(LineCollection(
            segments, colors="#60666D", linewidths=0.8, alpha=0.45,
            zorder=1))
        ax.scatter(
            sonar[:, 0], sonar[:, 1], s=17, c="#168F95", marker="o",
            alpha=0.65, linewidths=0, label="Sonar", zorder=2)
        ax.scatter(
            other[:, 0], other[:, 1], s=19, c=color, marker=marker,
            alpha=0.65, linewidths=0, label=label, zorder=2)
        ax.scatter(
            sonar[selected, 0], sonar[selected, 1], s=37, c="#168F95",
            marker="o", edgecolors="#333333", linewidths=0.5, zorder=3)
        ax.scatter(
            other[selected, 0], other[selected, 1], s=40, c=color,
            marker=marker, edgecolors="#333333", linewidths=0.5, zorder=3)
        ax.set(
            xlim=(low[0] - margin[0], high[0] + margin[0]),
            ylim=(low[1] - margin[1], high[1] + margin[1]),
            xlabel="t-SNE 1", ylabel="t-SNE 2",
            title="Sonar and {} | Validation, N={}".format(label, n))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", frameon=True, framealpha=0.92)
        for extension in (".png", ".pdf"):
            fig.savefig(
                args.out_dir / (filename + extension), dpi=220,
                bbox_inches="tight")
        plt.close(fig)

    metadata = {
        "source": str(args.source.resolve()),
        "features_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sample_count_per_modality": n,
        "joint_embedding_count": 3 * n,
        "parameters": parameters,
        "sklearn_version": sklearn.__version__,
        "kl_divergence": float(reducer.kl_divergence_),
        "linked_samples": [samples[int(index)] for index in selected],
        "notes": [
            "One joint fit for all three modalities; identical sonar coordinates "
            "and plot limits in both panels.",
            "Lines join 24 matched pairs sampled at seed 42, independently of "
            "feature values or projected distances.",
            "All validation features are plotted without manual relocation.",
            "Text features are RGB-conditioned outputs, not independent captions.",
            "Two-dimensional distances are approximate; use original cosine "
            "similarities for quantitative conclusions.",
        ],
    }
    (args.out_dir / "scatter_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done: {}".format(args.out_dir), flush=True)


if __name__ == "__main__":
    main()
