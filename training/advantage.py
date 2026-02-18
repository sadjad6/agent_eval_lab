"""Advantage estimation utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and value targets."""
    n = rewards.shape[0]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0.0

    for t in reversed(range(n)):
        next_value = 0.0 if t == n - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)
