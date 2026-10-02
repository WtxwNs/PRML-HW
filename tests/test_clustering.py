import numpy as np

from prml_project.clustering import DBSCAN, KMeans, LocalScaleSpectral


def test_kmeans_separates_two_compact_clusters():
    features = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    labels = KMeans(n_clusters=2, n_init=3, random_state=0).fit_predict(features)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_dbscan_marks_isolated_point_as_noise():
    features = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0]])
    labels = DBSCAN(eps=0.2, min_samples=3).fit_predict(features)
    assert len(set(labels[:3])) == 1
    assert labels[3] == -1


def test_local_scale_spectral_separates_compact_groups():
    rng = np.random.default_rng(0)
    features = np.vstack(
        [
            rng.normal(loc=-2.0, scale=0.1, size=(15, 2)),
            rng.normal(loc=2.0, scale=0.1, size=(15, 2)),
        ]
    )
    labels = LocalScaleSpectral(
        n_clusters=2,
        graph_neighbors=5,
        scale_neighbor=2,
        smoothing_neighbors=3,
        smoothing_steps=1,
        n_init=3,
    ).fit_predict(features)
    assert len(np.unique(labels)) == 2


def test_kmeans_refit_replaces_previous_dataset_state():
    model = KMeans(n_clusters=1, n_init=1, random_state=0)
    model.fit_predict(np.array([[0.0], [0.0]]))
    labels = model.fit_predict(np.array([[10.0], [20.0], [30.0]]))
    assert labels.shape == (3,)
    np.testing.assert_allclose(model.cluster_centers_, [[20.0]])
    assert model.inertia_ == 200.0
    assert model.n_iter_ > 0


def test_kmeans_integer_input_preserves_fractional_centers():
    model = KMeans(n_clusters=1, n_init=1, random_state=0)
    model.fit_predict(np.array([[0], [1]], dtype=np.int64))
    np.testing.assert_allclose(model.cluster_centers_, [[0.5]])
    assert model.inertia_ == 0.5


def test_kmeans_rejects_invalid_iteration_counts():
    for parameter in ("n_clusters", "n_init", "max_iter"):
        for value in (0, -1, 1.5):
            model = KMeans(**{parameter: value})
            try:
                model.fit_predict(np.array([[0.0], [1.0]]))
            except ValueError as error:
                assert parameter in str(error)
            else:
                raise AssertionError(f"{parameter}={value} should be rejected")
