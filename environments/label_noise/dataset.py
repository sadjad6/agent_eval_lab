"""Synthetic dataset with corrupted labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SplitData:
    """Container for split data."""

    features: np.ndarray
    labels: np.ndarray


class LabelNoiseDataset:
    """Generate train/validation splits with symmetric label noise on the train set."""

    def __init__(
        self,
        seed: int,
        num_samples: int = 2000,
        feature_dim: int = 16,
        num_classes: int = 3,
        train_ratio: float = 0.8,
        noise_rate: float = 0.2,
    ) -> None:
        self.seed = seed
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.train_ratio = train_ratio
        self.noise_rate = noise_rate
        self.rng = np.random.default_rng(seed)

        self.train_data, self.val_data = self._build_splits()

    def _build_splits(self) -> tuple[SplitData, SplitData]:
        train_n = int(self.num_samples * self.train_ratio)
        val_n = self.num_samples - train_n

        means = self.rng.normal(0.0, 1.5, size=(self.num_classes, self.feature_dim))
        cov = np.eye(self.feature_dim)
        priors = np.ones(self.num_classes, dtype=np.float64) / self.num_classes

        train_x, train_y = self._sample_clean(train_n, means, cov, priors)
        val_x, val_y = self._sample_clean(val_n, means, cov, priors)

        corrupt_mask = self.rng.random(train_n) < self.noise_rate
        random_labels = self.rng.choice(self.num_classes, size=int(np.sum(corrupt_mask)))
        train_y_corrupted = train_y.copy()
        train_y_corrupted[corrupt_mask] = random_labels

        self._clean_train_y = train_y

        return SplitData(train_x.astype(np.float32), train_y_corrupted.astype(np.int64)), SplitData(
            val_x.astype(np.float32), val_y.astype(np.int64)
        )

    def _sample_clean(
        self,
        n: int,
        means: np.ndarray,
        cov: np.ndarray,
        priors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = self.rng.choice(self.num_classes, size=n, p=priors)
        features = np.zeros((n, self.feature_dim), dtype=np.float64)
        for cls in range(self.num_classes):
            idx = np.where(labels == cls)[0]
            if idx.size == 0:
                continue
            features[idx] = self.rng.multivariate_normal(means[cls], cov, size=idx.size)
        return features, labels

    def get_split(self, split: str) -> SplitData:
        if split == "train":
            return self.train_data
        if split == "val":
            return self.val_data
        raise ValueError(f"Unknown split: {split}")

    def shift_metrics(self) -> Dict[str, float]:
        corrupted = int(np.sum(self.train_data.labels != self._clean_train_y))
        actual_rate = float(corrupted / len(self.train_data.labels))
        return {
            "target_noise_rate": self.noise_rate,
            "actual_noise_rate": actual_rate,
        }
