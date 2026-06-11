from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift


def standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return (features - mean) / scale


def deskew_images(images: np.ndarray) -> np.ndarray:
    """Correct horizontal shear estimated from second-order image moments."""
    corrected = np.empty_like(images, dtype=np.float64)
    yy, xx = np.indices(images.shape[1:])
    for index, image in enumerate(images):
        mass = image.sum()
        if mass <= 1e-12:
            corrected[index] = image
            continue
        center_y = float((yy * image).sum() / mass)
        center_x = float((xx * image).sum() / mass)
        covariance_xy = float(((xx - center_x) * (yy - center_y) * image).sum() / mass)
        variance_y = float((((yy - center_y) ** 2) * image).sum() / mass)
        shear = covariance_xy / max(variance_y, 1e-12)

        output = np.zeros_like(image, dtype=np.float64)
        for row in range(image.shape[0]):
            output[row] = shift(
                image[row],
                -shear * (row - center_y),
                order=1,
                mode="constant",
                prefilter=False,
            )
        corrected[index] = output
    return corrected


def extract_hog(
    images: np.ndarray,
    cells: tuple[int, int] = (2, 2),
    bins: int = 9,
) -> np.ndarray:
    """Compute a compact unsigned-gradient HOG descriptor for 8x8 digit images."""
    rows, cols = cells
    height, width = images.shape[1:]
    if height % rows or width % cols:
        raise ValueError("Image dimensions must be divisible by the HOG cell grid")

    gy, gx = np.gradient(images, axis=(1, 2))
    magnitude = np.hypot(gx, gy)
    orientation = np.mod(np.arctan2(gy, gx), np.pi)
    bin_index = np.minimum((orientation * bins / np.pi).astype(int), bins - 1)
    cell_h, cell_w = height // rows, width // cols
    descriptors = np.zeros((len(images), rows * cols * bins), dtype=np.float64)

    for row in range(rows):
        for col in range(cols):
            cell = row * cols + col
            row_slice = slice(row * cell_h, (row + 1) * cell_h)
            col_slice = slice(col * cell_w, (col + 1) * cell_w)
            for bin_id in range(bins):
                mask = bin_index[:, row_slice, col_slice] == bin_id
                descriptors[:, cell * bins + bin_id] = np.sum(
                    magnitude[:, row_slice, col_slice] * mask,
                    axis=(1, 2),
                )

    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return descriptors / np.maximum(norms, 1e-12)


@dataclass
class PCA:
    n_components: int
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    explained_variance_ratio_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "PCA":
        self.mean_ = features.mean(axis=0)
        centered = features - self.mean_
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        count = min(self.n_components, vt.shape[0])
        self.components_ = vt[:count]
        variance = singular_values**2 / max(len(features) - 1, 1)
        self.explained_variance_ratio_ = variance[:count] / variance.sum()
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA must be fitted before transform")
        return (features - self.mean_) @ self.components_.T

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)


def build_feature_sets(images: np.ndarray, pca_components: int) -> dict[str, np.ndarray]:
    pixels = standardize(images.reshape(len(images), -1))
    hog = standardize(extract_hog(images))
    pixel_pca = standardize(PCA(pca_components).fit_transform(pixels))
    hog_pca = standardize(PCA(min(pca_components, hog.shape[1])).fit_transform(hog))
    return {"pixel": pixels, "pixel_pca": pixel_pca, "hog": hog, "hog_pca": hog_pca}


def fuse_features(pixel_features: np.ndarray, hog_features: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    return np.concatenate((alpha * pixel_features, (1.0 - alpha) * hog_features), axis=1)
