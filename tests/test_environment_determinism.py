"""Tests for deterministic environment behavior."""

from __future__ import annotations

import unittest

import numpy as np

from environments.retrieval_shift.dataset import RetrievalShiftDataset
from environments.retrieval_shift.env import RetrievalShiftEnv


class TestEnvironmentDeterminism(unittest.TestCase):
    def test_environment_is_deterministic_under_seed(self) -> None:
        dataset1 = RetrievalShiftDataset(seed=7)
        dataset2 = RetrievalShiftDataset(seed=7)

        env1 = RetrievalShiftEnv(dataset1, "train", seed=7, entropy_penalty_coef=0.1, kl_penalty_coef=0.1, kl_threshold=0.1)
        env2 = RetrievalShiftEnv(dataset2, "train", seed=7, entropy_penalty_coef=0.1, kl_penalty_coef=0.1, kl_threshold=0.1)

        traj1 = []
        traj2 = []
        probs = np.ones(dataset1.num_classes, dtype=np.float64) / dataset1.num_classes

        for _ in range(20):
            obs1 = env1.reset()
            _, r1, _, _ = env1.step({"action": 0, "probs": probs})
            traj1.append((obs1.copy(), float(r1)))

            obs2 = env2.reset()
            _, r2, _, _ = env2.step({"action": 0, "probs": probs})
            traj2.append((obs2.copy(), float(r2)))

        for (o1, r1), (o2, r2) in zip(traj1, traj2):
            self.assertTrue(np.allclose(o1, o2))
            self.assertAlmostEqual(r1, r2, places=7)


if __name__ == "__main__":
    unittest.main()
