"""Retrieval environment under controlled distribution shift."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.env_interface import EnvInterface
from environments.retrieval_shift.dataset import RetrievalShiftDataset


class RetrievalShiftEnv(EnvInterface):
    """One-step contextual bandit environment for retrieval policy training."""

    def __init__(
        self,
        dataset: RetrievalShiftDataset,
        split: str,
        seed: int,
        entropy_penalty_coef: float,
        kl_penalty_coef: float,
        kl_threshold: float,
        top_k: int = 1,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.entropy_penalty_coef = entropy_penalty_coef
        self.kl_penalty_coef = kl_penalty_coef
        self.kl_threshold = kl_threshold
        self.top_k = max(1, top_k)

        self.data = dataset.get_split(split)
        self.num_classes = dataset.num_classes
        self.feature_dim = dataset.feature_dim

        self._order = self.rng.permutation(len(self.data.labels))
        self._cursor = 0

        self._obs: np.ndarray | None = None
        self._label: int | None = None
        self._action_probs: np.ndarray | None = None
        self._last_kl = 0.0
        self._last_entropy = 0.0

        self._metrics: Dict[str, list[float]] = {
            "reward": [],
            "accuracy": [],
            "entropy": [],
            "kl": [],
        }

        init_rng = np.random.default_rng(seed + 17)
        self._init_w = init_rng.normal(0.0, 0.15, size=(self.feature_dim, self.num_classes))
        self._init_b = init_rng.normal(0.0, 0.05, size=(self.num_classes,))

    def _next_index(self) -> int:
        if self._cursor >= len(self._order):
            self._order = self.rng.permutation(len(self.data.labels))
            self._cursor = 0
        idx = int(self._order[self._cursor])
        self._cursor += 1
        return idx

    def reset(self) -> np.ndarray:
        idx = self._next_index()
        self._obs = self.data.features[idx]
        self._label = int(self.data.labels[idx])
        return self._obs.copy()

    def _initial_policy_probs(self, obs: np.ndarray) -> np.ndarray:
        logits = obs @ self._init_w + self._init_b
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self._obs is None or self._label is None:
            raise RuntimeError("reset() must be called before step().")

        if isinstance(action, dict):
            action_idx = int(action["action"])
            probs = np.asarray(action.get("probs"), dtype=np.float64)
            if probs.ndim != 1 or probs.shape[0] != self.num_classes:
                raise ValueError("Action probs must be a 1D vector of num_classes.")
        else:
            action_idx = int(action)
            probs = np.zeros(self.num_classes, dtype=np.float64)
            probs[action_idx] = 1.0

        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum()
        self._action_probs = probs

        reward = self.compute_reward(action_idx)
        done = True
        next_obs = self.reset()

        info = {
            "label": self._label,
            "entropy": self._last_entropy,
            "kl_to_initial": self._last_kl,
            "top_k": self.top_k,
        }
        return next_obs, reward, done, info

    def compute_reward(self, action: Any) -> float:
        if self._obs is None or self._label is None or self._action_probs is None:
            raise RuntimeError("Action probabilities and current sample are required.")

        _ = int(action)
        topk_idx = np.argsort(self._action_probs)[-self.top_k :]
        correct = 1.0 if self._label in set(topk_idx.tolist()) else 0.0

        entropy = -float(np.sum(self._action_probs * np.log(self._action_probs + 1e-8)))
        max_entropy = float(np.log(self.num_classes))
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.0
        entropy_penalty = self.entropy_penalty_coef * max(0.0, 0.3 - entropy_ratio)

        init_probs = self._initial_policy_probs(self._obs)
        kl = float(np.sum(self._action_probs * (np.log(self._action_probs + 1e-8) - np.log(init_probs + 1e-8))))
        kl_penalty = self.kl_penalty_coef * max(0.0, kl - self.kl_threshold)

        reward = correct - entropy_penalty - kl_penalty

        self._last_entropy = entropy
        self._last_kl = kl
        self._metrics["reward"].append(float(reward))
        self._metrics["accuracy"].append(correct)
        self._metrics["entropy"].append(entropy)
        self._metrics["kl"].append(kl)

        return float(reward)

    def get_metrics(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for key, values in self._metrics.items():
            result[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
            result[f"{key}_var"] = float(np.var(values)) if values else 0.0
        return result
