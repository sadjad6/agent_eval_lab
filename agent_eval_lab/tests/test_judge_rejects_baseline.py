"""Tests for judge model-type enforcement."""

from __future__ import annotations

import os
import tempfile
import unittest

import torch

from environments.retrieval_shift.judge import RetrievalShiftJudge


class TestJudgeRejectsBaseline(unittest.TestCase):
    def test_rejects_cosine_only_baseline(self) -> None:
        cfg = {
            "seed": 123,
            "dataset": {"num_samples": 1500, "feature_dim": 32, "num_classes": 5, "train_ratio": 0.8},
            "ood_thresholds": {
                "ood_accuracy_min": 0.30,
                "entropy_min": 0.8,
                "gradient_norm_max": 5.0,
                "generalization_gap_max": 0.3,
                "retrieval_kl_max": 1.0,
            },
        }

        with tempfile.TemporaryDirectory() as td:
            bad_model = os.path.join(td, "baseline.pt")
            torch.save(
                {
                    "obs_dim": 32,
                    "num_actions": 5,
                    "model_state_dict": {},
                    "metadata": {"model_type": "cosine_baseline"},
                },
                bad_model,
            )

            judge = RetrievalShiftJudge(cfg)
            result = judge.evaluate(bad_model)

            self.assertFalse(result.passed)
            self.assertLessEqual(result.score, 0.01)
            self.assertIn("rejected", result.message.lower())


if __name__ == "__main__":
    unittest.main()
