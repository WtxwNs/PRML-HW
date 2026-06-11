import numpy as np

from prml_project.metrics import evaluate_clustering


def test_dbscan_noise_counts_as_full_accuracy_error():
    features = np.array([[0.0], [0.1], [1.0], [1.1]])
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([3, 3, 4, -1])
    metrics = evaluate_clustering(features, y_true, y_pred)
    assert metrics["conditional_acc"] == 1.0
    assert metrics["coverage"] == 0.75
    assert metrics["acc"] == 0.75
