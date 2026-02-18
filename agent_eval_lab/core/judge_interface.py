"""Judge interface for deterministic scoring of trained agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict


@dataclass
class JudgeResult:
    """Structured judge output."""

    score: float
    passed: bool
    metrics: Dict[str, float]
    checks: Dict[str, bool]
    message: str


class JudgeInterface(ABC):
    """Base interface for all deterministic judges."""

    @abstractmethod
    def evaluate(self, model_path: str) -> JudgeResult:
        """Evaluate a saved model and return a normalized judge result."""
