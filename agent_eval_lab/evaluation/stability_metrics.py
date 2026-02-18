"""Stability checks and rolling statistics."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def rolling_mean(values: Sequence[float], window: int) -> float:
    """Compute rolling mean over the most recent values."""
    if not values:
        return 0.0
    if len(values) < window:
        return float(np.mean(values))
    return float(np.mean(values[-window:]))


def gradient_spike(current: float, history: Sequence[float], factor: float = 5.0, window: int = 20) -> bool:
    """Detect abnormal gradient norm spikes."""
    baseline = rolling_mean(history, window)
    if baseline <= 1e-8:
        return False
    return current > factor * baseline
