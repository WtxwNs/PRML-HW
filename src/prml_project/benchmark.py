from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import silhouette_score

from .clustering import DBSCAN, KMeans
from .data import load_digits_dataset
from .features import PCA, build_feature_sets, fuse_features, standardize
from .metrics import evaluate_clustering


def _run_model(name, features, labels, model) -> dict[str, float | str]:
    started = time.perf_counter()
    predictions = model.fit_predict(features)
    elapsed = time.perf_counter() - started
    result: dict[str, float | str] = {"experiment": name, "seconds": elapsed}
    result.update(evaluate_clustering(features, labels, predictions))
    if isinstance(model, KMeans):
        result["iterations"] = float(model.n_iter_)
        result["inertia"] = model.inertia_
    return result


def run_benchmark(config: dict) -> pd.DataFrame:
    seed = int(config["seed"])
    dataset = load_digits_dataset(config.get("max_samples"), seed)
    feature_sets = build_feature_sets(dataset.images, int(config["pca_components"]))
    kmeans_config = config["kmeans"]
    results = []

    for name, features, init in [
        ("pixel_random_kmeans", feature_sets["pixel"], "random"),
        ("pixel_pca_random_kmeans", feature_sets["pixel_pca"], "random"),
        ("hog_kmeans++", feature_sets["hog"], "kmeans++"),
    ]:
        model = KMeans(init=init, random_state=seed, **kmeans_config)
        results.append(_run_model(name, features, dataset.labels, model))

    fusion_candidates = []
    for alpha in config["fusion_alphas"]:
        fused = fuse_features(feature_sets["pixel_pca"], feature_sets["hog_pca"], float(alpha))
        fused = standardize(PCA(int(config["pca_components"])).fit_transform(fused))
        model = KMeans(init="kmeans++", random_state=seed, **kmeans_config)
        started = time.perf_counter()
        predictions = model.fit_predict(fused)
        elapsed = time.perf_counter() - started
        score = silhouette_score(fused, predictions)
        fusion_candidates.append((score, float(alpha), fused, model, predictions, elapsed))
    _, alpha, fused, model, predictions, elapsed = max(
        fusion_candidates, key=lambda item: item[0]
    )
    fusion_result = evaluate_clustering(fused, dataset.labels, predictions)
    fusion_result.update(
        {
            "experiment": "adaptive_fusion_kmeans++",
            "alpha": alpha,
            "seconds": elapsed,
            "iterations": float(model.n_iter_),
            "inertia": model.inertia_,
        }
    )
    results.append(fusion_result)

    dbscan_config = config["dbscan"]
    for eps in dbscan_config["eps_values"]:
        for min_samples in dbscan_config["min_samples_values"]:
            name = f"dbscan_eps={eps}_min={min_samples}"
            model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
            result = _run_model(name, feature_sets["hog_pca"], dataset.labels, model)
            result.update({"eps": float(eps), "min_samples": float(min_samples)})
            results.append(result)

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the handwritten digit clustering benchmark")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    results = run_benchmark(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark_results.csv"
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
