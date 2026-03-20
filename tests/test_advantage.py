"""Tests for advantage estimation."""

from __future__ import annotations

import unittest

import numpy as np

from training.advantage import compute_gae


class TestAdvantage(unittest.TestCase):
    def test_one_step_fast_path(self) -> None:
        rewards = np.array([1.0, 0.0, 1.0])
        values = np.array([0.5, 0.2, 0.8])
        dones = np.array([1.0, 1.0, 1.0])
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
        np.testing.assert_allclose(adv, rewards - values)
        np.testing.assert_allclose(ret, rewards)

    def test_multi_step(self) -> None:
        rewards = np.array([1.0, 0.0])
        values = np.array([0.5, 0.2])
        dones = np.array([0.0, 1.0])
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
        self.assertAlmostEqual(adv[1], -0.2, places=4)
        self.assertAlmostEqual(ret[1], 0.0, places=4)
        self.assertAlmostEqual(adv[0], 0.5099, places=4)
        self.assertAlmostEqual(ret[0], 1.0099, places=4)


if __name__ == "__main__":
    unittest.main()
