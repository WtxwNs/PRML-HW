from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from .clustering import DBSCAN, KMeans, LocalScaleSpectral
from .data import load_digits_dataset
from .features import PCA, deskew_images, extract_hog, standardize
from .metrics import evaluate_clustering

sns.set_theme(style="whitegrid", context="paper")


def _evaluate(name: str, features: np.ndarray, labels: np.ndarray, predictions: np.ndarray, elapsed):
    row: dict[str, float | str] = {"method": name, "seconds": elapsed}
    row.update(evaluate_clustering(features, labels, predictions))
    return row


def _timed_predict(model, features: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    predictions = model.fit_predict(features)
    return predictions, time.perf_counter() - started


def _matched_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)
    counts = np.array(
        [[np.sum((y_pred == pred) & (y_true == true)) for true in true_values] for pred in pred_values]
    )
    rows, cols = linear_sum_assignment(counts.max() - counts)
    mapping = {pred_values[row]: true_values[col] for row, col in zip(rows, cols)}
    return np.array([mapping.get(value, -1) for value in y_pred], dtype=np.int64)


def _run_main_benchmark(raw, deskewed, labels, config):
    seed = int(config["seed"])
    n_init = int(config["kmeans_n_init"])
    pca = PCA(int(config["pca_components"])).fit_transform(raw)
    hog = standardize(extract_hog(raw.reshape(-1, 8, 8)))
    rows = []
    predictions = {}

    models = [
        ("K-means (raw)", raw, KMeans(n_init=n_init, random_state=seed)),
        ("K-means++ (PCA)", pca, KMeans(init="kmeans++", n_init=n_init, random_state=seed)),
        ("K-means++ (HOG)", hog, KMeans(init="kmeans++", n_init=n_init, random_state=seed)),
        (
            "DBSCAN (PCA)",
            pca,
            DBSCAN(
                eps=float(config["dbscan"]["eps"]),
                min_samples=int(config["dbscan"]["min_samples"]),
            ),
        ),
        (
            "Gaussian mixture",
            pca,
            GaussianMixture(
                n_components=10,
                covariance_type="full",
                n_init=5,
                random_state=seed,
            ),
        ),
        (
            "Standard spectral",
            raw,
            SpectralClustering(
                n_clusters=10,
                affinity="nearest_neighbors",
                n_neighbors=int(config["standard_spectral_neighbors"]),
                assign_labels="kmeans",
                n_init=n_init,
                random_state=seed,
            ),
        ),
        (
            "Proposed method",
            deskewed,
            LocalScaleSpectral(
                graph_neighbors=int(config["proposed"]["graph_neighbors"]),
                scale_neighbor=int(config["proposed"]["scale_neighbor"]),
                smoothing_neighbors=int(config["proposed"]["smoothing_neighbors"]),
                smoothing_steps=int(config["proposed"]["smoothing_steps"]),
                n_init=n_init,
                random_state=seed,
            ),
        ),
    ]

    for name, features, model in models:
        pred, elapsed = _timed_predict(model, features)
        predictions[name] = pred
        rows.append(_evaluate(name, features, labels, pred, elapsed))
    return pd.DataFrame(rows), predictions


def _run_ablation(raw, deskewed, labels, config):
    proposed = config["proposed"]
    seed = int(config["seed"])
    n_init = int(config["kmeans_n_init"])
    variants = [
        ("A0: raw K-means++", raw, None),
        ("A1: deskew + K-means++", deskewed, None),
        ("A2: local-scale spectral", raw, 0),
        ("A3: + deskew", deskewed, 0),
        ("A4: + neighborhood refinement", deskewed, int(proposed["smoothing_steps"])),
    ]
    rows = []
    for name, features, smoothing_steps in variants:
        model = (
            KMeans(init="kmeans++", n_init=n_init, random_state=seed)
            if smoothing_steps is None
            else LocalScaleSpectral(
                graph_neighbors=int(proposed["graph_neighbors"]),
                scale_neighbor=int(proposed["scale_neighbor"]),
                smoothing_neighbors=int(proposed["smoothing_neighbors"]),
                smoothing_steps=smoothing_steps,
                n_init=n_init,
                random_state=seed,
            )
        )
        pred, elapsed = _timed_predict(model, features)
        rows.append(_evaluate(name, features, labels, pred, elapsed))
    return pd.DataFrame(rows)


def _run_sensitivity(deskewed, labels, config):
    proposed = config["proposed"]
    rows = []
    for graph_neighbors in config["sensitivity"]["graph_neighbors"]:
        for scale_neighbor in config["sensitivity"]["scale_neighbors"]:
            if scale_neighbor > graph_neighbors:
                continue
            model = LocalScaleSpectral(
                graph_neighbors=int(graph_neighbors),
                scale_neighbor=int(scale_neighbor),
                smoothing_neighbors=int(proposed["smoothing_neighbors"]),
                smoothing_steps=int(proposed["smoothing_steps"]),
                n_init=int(config["sensitivity"]["n_init"]),
                random_state=int(config["seed"]),
            )
            pred, elapsed = _timed_predict(model, deskewed)
            row = _evaluate("sensitivity", deskewed, labels, pred, elapsed)
            row.update(
                {
                    "graph_neighbors": int(graph_neighbors),
                    "scale_neighbor": int(scale_neighbor),
                    "modularity": model.modularity_,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _run_stability(deskewed, labels, config):
    proposed = config["proposed"]
    rows = []
    for seed in config["stability_seeds"]:
        model = LocalScaleSpectral(
            graph_neighbors=int(proposed["graph_neighbors"]),
            scale_neighbor=int(proposed["scale_neighbor"]),
            smoothing_neighbors=int(proposed["smoothing_neighbors"]),
            smoothing_steps=int(proposed["smoothing_steps"]),
            n_init=int(config["kmeans_n_init"]),
            random_state=int(seed),
        )
        pred, elapsed = _timed_predict(model, deskewed)
        row = _evaluate("Proposed method", deskewed, labels, pred, elapsed)
        row["seed"] = int(seed)
        rows.append(row)
    return pd.DataFrame(rows)


def _save_plots(output_dir, raw_images, deskewed_images, labels, benchmark, ablation, sensitivity, predictions):
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for col in range(8):
        axes[0, col].imshow(raw_images[col], cmap="gray")
        axes[1, col].imshow(deskewed_images[col], cmap="gray")
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("Raw")
    axes[1, 0].set_ylabel("Deskewed")
    fig.tight_layout()
    fig.savefig(figures / "deskew_examples.pdf", bbox_inches="tight")
    plt.close(fig)

    plot_data = benchmark.melt(
        id_vars="method", value_vars=["acc", "nmi", "ari"], var_name="metric", value_name="score"
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=plot_data, x="method", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figures / "benchmark_metrics.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=ablation, x="method", y="acc", color="#4c72b0", ax=ax)
    ax.set_ylim(0.7, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("ACC")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures / "ablation_acc.pdf", bbox_inches="tight")
    plt.close(fig)

    table = sensitivity.pivot(
        index="scale_neighbor", columns="graph_neighbors", values="acc"
    )
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.heatmap(table, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.8, vmax=0.93, ax=ax)
    ax.set_xlabel("Graph neighbors")
    ax.set_ylabel("Local scale neighbor")
    fig.tight_layout()
    fig.savefig(figures / "sensitivity_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    for method in ["K-means (raw)", "Proposed method"]:
        matched = _matched_predictions(labels, predictions[method])
        matrix = confusion_matrix(labels, matched, labels=np.arange(10), normalize="true")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
        ax.set_xlabel("Matched cluster label")
        ax.set_ylabel("True label")
        fig.tight_layout()
        filename = "confusion_" + ("baseline" if method.startswith("K-means") else "proposed")
        fig.savefig(figures / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)

    distances, _ = NearestNeighbors(n_neighbors=6).fit(deskewed_images.reshape(len(labels), -1)).kneighbors()
    curve = np.sort(distances[:, -1])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(curve)
    ax.set_xlabel("Samples sorted by 5-NN distance")
    ax.set_ylabel("5-NN distance")
    fig.tight_layout()
    fig.savefig(figures / "k_distance_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def run(config: dict):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_digits_dataset()
    raw_images = dataset.images
    deskewed_images = deskew_images(raw_images)
    raw = raw_images.reshape(len(dataset.labels), -1)
    deskewed = deskewed_images.reshape(len(dataset.labels), -1)

    benchmark, predictions = _run_main_benchmark(raw, deskewed, dataset.labels, config)
    ablation = _run_ablation(raw, deskewed, dataset.labels, config)
    sensitivity = _run_sensitivity(deskewed, dataset.labels, config)
    stability = _run_stability(deskewed, dataset.labels, config)

    benchmark.to_csv(output_dir / "benchmark.csv", index=False)
    ablation.to_csv(output_dir / "ablation.csv", index=False)
    sensitivity.to_csv(output_dir / "sensitivity.csv", index=False)
    stability.to_csv(output_dir / "stability.csv", index=False)
    summary = {
        "stability_mean": stability[["acc", "nmi", "ari"]].mean().to_dict(),
        "stability_std": stability[["acc", "nmi", "ari"]].std(ddof=0).to_dict(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_plots(
        output_dir,
        raw_images,
        deskewed_images,
        dataset.labels,
        benchmark,
        ablation,
        sensitivity,
        predictions,
    )
    return benchmark, ablation, sensitivity, stability


def main():
    parser = argparse.ArgumentParser(description="Run full course-project experiments")
    parser.add_argument("--config", type=Path, default=Path("configs/final_experiments.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    benchmark, ablation, _, stability = run(config)
    print("\nBenchmark\n", benchmark.to_string(index=False))
    print("\nAblation\n", ablation.to_string(index=False))
    print("\nStability mean\n", stability[["acc", "nmi", "ari"]].mean().to_string())


if __name__ == "__main__":
    main()
