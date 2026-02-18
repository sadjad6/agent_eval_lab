"""Tests for distribution shift construction."""

from __future__ import annotations

import unittest

import numpy as np

from environments.retrieval_shift.dataset import RetrievalShiftDataset


class TestDatasetShift(unittest.TestCase):
    def test_covariance_shift_exists(self) -> None:
        dataset = RetrievalShiftDataset(seed=42)
        train = dataset.get_split("train").features
        val = dataset.get_split("val").features

        cov_train = np.cov(train, rowvar=False)
        cov_val = np.cov(val, rowvar=False)
        delta = np.linalg.norm(cov_train - cov_val, ord="fro")
        self.assertGreater(delta, 1.0)


if __name__ == "__main__":
    unittest.main()
