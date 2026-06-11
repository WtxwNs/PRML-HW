from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_values = np.unique(y_true)
    pred_values = np.unique(y_pred)
    matrix = np.zeros((len(pred_values), len(true_values)), dtype=np.int64)
    for row, pred in enumerate(pred_values):
        for col, true in enumerate(true_values):
            matrix[row, col] = np.sum((y_pred == pred) & (y_true == true))
    rows, cols = linear_sum_assignment(matrix.max() - matrix)
    return float(matrix[rows, cols].sum() / len(y_true))


def evaluate_clustering(
    features: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    non_noise = y_pred >= 0
    coverage = float(non_noise.mean())
    conditional_acc = (
        clustering_accuracy(y_true[non_noise], y_pred[non_noise]) if non_noise.any() else 0.0
    )
    full_acc = conditional_acc * coverage
    unique_non_noise = np.unique(y_pred[non_noise])
    silhouette = (
        float(silhouette_score(features[non_noise], y_pred[non_noise]))
        if len(unique_non_noise) >= 2 and non_noise.sum() > len(unique_non_noise)
        else float("nan")
    )
    return {
        "acc": full_acc,
        "conditional_acc": conditional_acc,
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "silhouette": silhouette,
        "coverage": coverage,
        "noise_ratio": 1.0 - coverage,
        "n_clusters": float(len(unique_non_noise)),
    }
