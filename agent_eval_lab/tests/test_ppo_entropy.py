"""Tests for PPO trainer entropy behavior."""

from __future__ import annotations

import shutil
import tempfile
import unittest

from environments.retrieval_shift.dataset import RetrievalShiftDataset
from environments.retrieval_shift.env import RetrievalShiftEnv
from training.ppo_trainer import PPOTrainer


class TestPPOEntropy(unittest.TestCase):
    def test_trainer_produces_non_zero_entropy(self) -> None:
        out_dir = tempfile.mkdtemp(prefix="ppo_entropy_test_")
        try:
            cfg = {
                "seed": 11,
                "dataset": {"num_samples": 1500, "feature_dim": 32, "num_classes": 5, "train_ratio": 0.8},
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
                    "hidden_dim": 64,
                    "output_dir": out_dir,
                },
            }
            dataset = RetrievalShiftDataset(seed=cfg["seed"])
            env = RetrievalShiftEnv(
                dataset=dataset,
                split="train",
                seed=cfg["seed"],
                entropy_penalty_coef=0.1,
                kl_penalty_coef=0.1,
                kl_threshold=0.1,
            )
            trainer = PPOTrainer(env, cfg)
            metrics = trainer.train()
            self.assertGreater(metrics["policy_entropy"], 0.0)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
