"""Logging utilities for training and evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict


def configure_logging(log_level: str = "INFO") -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class MetricsLogger:
    """JSONL metrics sink for CI and analysis tooling."""

    def __init__(self, log_path: str) -> None:
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
