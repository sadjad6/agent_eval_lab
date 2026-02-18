"""Core environment interface for deterministic RL experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class EnvInterface(ABC):
    """Base interface for all environments in this project."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""

    @abstractmethod
    def step(self, action: Any) -> tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply an action and return (next_obs, reward, done, info)."""

    @abstractmethod
    def compute_reward(self, action: Any) -> float:
        """Compute reward for the current transition."""

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return aggregate environment metrics."""
