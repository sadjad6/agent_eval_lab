"""Synthetic retrieval dataset with controlled distribution shift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SplitData:
    """Container for split data."""

    features: np.ndarray
    labels: np.ndarray


class RetrievalShiftDataset:
    """Generate train/validation splits with class-prior and covariance shift."""

    def __init__(
        self,
        seed: int,
        num_samples: int = 1500,
        feature_dim: int = 32,
        num_classes: int = 5,
        train_ratio: float = 0.8,
    ) -> None:
        self.seed = seed
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.train_ratio = train_ratio
        self.rng = np.random.default_rng(seed)

        self.train_data, self.val_data = self._build_splits()

    def _build_splits(self) -> tuple[SplitData, SplitData]:
        train_n = int(self.num_samples * self.train_ratio)
        val_n = self.num_samples - train_n

        means = self.rng.normal(0.0, 1.5, size=(self.num_classes, self.feature_dim))

        base_scales = self.rng.uniform(0.5, 1.5, size=self.feature_dim)
        train_cov = np.diag(base_scales)

        shift_matrix = self.rng.normal(0.0, 0.05, size=(self.feature_dim, self.feature_dim))
        transform = np.eye(self.feature_dim) + shift_matrix
        val_cov = transform @ train_cov @ transform.T
        val_cov += 0.2 * np.eye(self.feature_dim)

        train_priors = np.ones(self.num_classes, dtype=np.float64)
        train_priors /= train_priors.sum()

        raw_val_priors = np.array([0.50, 0.20, 0.15, 0.10, 0.05], dtype=np.float64)
        if self.num_classes != 5:
            raw_val_priors = self.rng.uniform(0.1, 1.0, size=self.num_classes)
        val_priors = raw_val_priors / raw_val_priors.sum()

        train_x, train_y = self._sample_split(train_n, means, train_cov, train_priors)
        val_x, val_y = self._sample_split(val_n, means, val_cov, val_priors)

        self._train_cov = np.cov(train_x, rowvar=False)
        self._val_cov = np.cov(val_x, rowvar=False)
        self._train_priors = train_priors
        self._val_priors = val_priors

        return SplitData(train_x.astype(np.float32), train_y.astype(np.int64)), SplitData(
            val_x.astype(np.float32), val_y.astype(np.int64)
        )

    def _sample_split(
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
        cov_delta = float(np.linalg.norm(self._train_cov - self._val_cov, ord="fro"))
        prior_delta = float(np.linalg.norm(self._train_priors - self._val_priors, ord=1))
        return {
            "covariance_frobenius_delta": cov_delta,
            "class_prior_l1_delta": prior_delta,
        }
