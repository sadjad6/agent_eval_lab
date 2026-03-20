"""OOD evaluation for retrieval shift policy checkpoints."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from core.seed import set_global_seed
from environments.retrieval_shift.dataset import RetrievalShiftDataset
from training.ppo_trainer import PolicyValueNet


def _evaluate_split(model: PolicyValueNet, features: np.ndarray, labels: np.ndarray, device: str = "cpu") -> Dict[str, Any]:
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(labels).to(device)
    with torch.no_grad():
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        acc = float((pred == y).float().mean().item())
        action_dist = probs.mean(dim=0).cpu().numpy().astype(np.float64)
    return {"accuracy": acc, "distribution": action_dist}


from core.config_schema import PipelineConfig

def evaluate_ood(model_path: str, config: PipelineConfig, device: str = "cpu") -> Dict[str, float]:
    """Load checkpoint and evaluate train/val distribution shift sensitivity."""
    set_global_seed(int(config.seed))

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = PolicyValueNet(
        obs_dim=int(ckpt["obs_dim"]),
        num_actions=int(ckpt["num_actions"]),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    dcfg = config.dataset
    dataset = RetrievalShiftDataset(
        seed=int(config.seed),
        num_samples=int(dcfg.num_samples),
        feature_dim=int(dcfg.feature_dim),
        num_classes=int(dcfg.num_classes),
        train_ratio=float(dcfg.train_ratio),
    )

    train = dataset.get_split("train")
    val = dataset.get_split("val")
    train_m = _evaluate_split(model, train.features, train.labels, device)
    val_m = _evaluate_split(model, val.features, val.labels, device)

    p = train_m["distribution"]
    q = val_m["distribution"]
    retrieval_kl = float(np.sum(p * (np.log(p + 1e-8) - np.log(q + 1e-8))))
    generalization_gap = float(train_m["accuracy"] - val_m["accuracy"])

    return {
        "train_accuracy": float(train_m["accuracy"]),
        "val_accuracy": float(val_m["accuracy"]),
        "generalization_gap": generalization_gap,
        "retrieval_distribution_kl": retrieval_kl,
    }
