"""Tests for judge model-type enforcement."""

from __future__ import annotations

import os
import tempfile
import unittest

import torch

from environments.retrieval_shift.judge import RetrievalShiftJudge


from core.config_schema import PipelineConfig


class TestJudgeRejectsBaseline(unittest.TestCase):
    def test_rejects_cosine_only_baseline(self) -> None:
        cfg = PipelineConfig.from_dict({
            "seed": 123,
            "dataset": {"num_samples": 1500, "feature_dim": 32, "num_classes": 5, "train_ratio": 0.8},
            "env": {
                "name": "retrieval_shift",
                "entropy_penalty_coef": 0.1,
                "kl_penalty_coef": 0.1,
                "kl_threshold": 0.1,
                "top_k": 1,
            },
            "training": {
                "learning_rate": 0.001,
                "clip_epsilon": 0.2,
                "entropy_coef": 0.01,
                "value_coef": 0.5,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "max_grad_norm": 1.0,
                "train_iterations": 2,
                "steps_per_iter": 64,
                "update_epochs": 2,
                "minibatch_size": 32,
                "kl_abort_threshold": 1.0,
                "hidden_dim": 128,
                "output_dir": "outputs/test",
                "device": "cpu",
            },
            "ood_thresholds": {
                "ood_accuracy_min": 0.30,
                "entropy_min": 0.8,
                "gradient_norm_max": 5.0,
                "generalization_gap_max": 0.3,
                "retrieval_kl_max": 1.0,
            },
            "reproducibility": {"dual_run_test": False}
        })

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
