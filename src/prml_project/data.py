from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits


@dataclass(frozen=True)
class DigitsDataset:
    images: np.ndarray
    features: np.ndarray
    labels: np.ndarray


def load_digits_dataset(max_samples: int | None = None, seed: int = 42) -> DigitsDataset:
    dataset = load_digits()
    images = dataset.images.astype(np.float64) / 16.0
    labels = dataset.target.astype(np.int64)

    if max_samples is not None and max_samples < len(labels):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(labels), size=max_samples, replace=False)
        images = images[indices]
        labels = labels[indices]

    return DigitsDataset(images=images, features=images.reshape(len(images), -1), labels=labels)

