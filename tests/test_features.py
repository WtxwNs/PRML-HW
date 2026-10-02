import numpy as np

from prml_project.features import PCA, deskew_images, extract_hog, fuse_features


def test_hog_shape_and_finite_values():
    images = np.zeros((3, 8, 8), dtype=float)
    descriptors = extract_hog(images)
    assert descriptors.shape == (3, 36)
    assert np.isfinite(descriptors).all()


def test_pca_and_fusion_shapes():
    rng = np.random.default_rng(0)
    left = rng.normal(size=(20, 8))
    right = rng.normal(size=(20, 4))
    reduced = PCA(3).fit_transform(left)
    fused = fuse_features(reduced, right, 0.5)
    assert reduced.shape == (20, 3)
    assert fused.shape == (20, 7)


def test_deskew_preserves_shape_and_finite_values():
    images = np.zeros((2, 8, 8), dtype=float)
    images[0, 1:7, 3] = 1.0
    corrected = deskew_images(images)
    assert corrected.shape == images.shape
    assert np.isfinite(corrected).all()


def test_pca_constant_input_has_finite_zero_variance_ratios():
    model = PCA(2)
    reduced = model.fit_transform(np.ones((4, 3)))
    np.testing.assert_allclose(reduced, 0.0)
    np.testing.assert_array_equal(model.explained_variance_ratio_, [0.0, 0.0])
