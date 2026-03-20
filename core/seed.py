"""Global seeding and reproducibility utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def config_hash(config: Dict[str, Any]) -> str:
    """Create a stable hash for config tracking."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
