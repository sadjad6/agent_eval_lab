"""Deterministic judge implementation for retrieval shift environment."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core.judge_interface import JudgeInterface, JudgeResult
from core.seed import set_global_seed
from environments.retrieval_shift.dataset import RetrievalShiftDataset
from training.ppo_trainer import PolicyValueNet


class RetrievalShiftJudge(JudgeInterface):
    """Execution-based deterministic judge with thresholded quality checks."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def _load_model(self, model_path: str) -> Tuple[PolicyValueNet, Dict[str, Any]]:
        payload = torch.load(model_path, map_location="cpu")
        metadata = payload.get("metadata", {})

        if metadata.get("model_type") != "mlp_policy":
            raise ValueError("Unsupported model type for judge.")

        model = PolicyValueNet(
            obs_dim=int(payload["obs_dim"]),
            num_actions=int(payload["num_actions"]),
            hidden_dim=int(payload.get("hidden_dim", 128)),
        )
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model, metadata

    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

    def _evaluate_split(
        self,
        model: PolicyValueNet,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        x = torch.from_numpy(features)
        y = torch.from_numpy(labels)
        with torch.no_grad():
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            acc = float((pred == y).float().mean().item())
            entropy = float(self._entropy_from_logits(logits).mean().item())
        return {
            "accuracy": acc,
            "entropy": entropy,
            "action_distribution": probs.mean(dim=0).cpu().numpy().astype(np.float64),
        }

    def _gradient_norm(self, model: PolicyValueNet, features: np.ndarray, labels: np.ndarray) -> float:
        model.zero_grad(set_to_none=True)
        x = torch.from_numpy(features[:128])
        y = torch.from_numpy(labels[:128])
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        grad_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_sq += float(torch.sum(p.grad.detach() ** 2).item())
        model.zero_grad(set_to_none=True)
        return float(np.sqrt(grad_sq))

    def evaluate(self, model_path: str) -> JudgeResult:
        seed = int(self.config["seed"])
        set_global_seed(seed)

        if not os.path.exists(model_path):
            return JudgeResult(
                score=0.0,
                passed=False,
                metrics={},
                checks={"model_exists": False},
                message="Model file does not exist.",
            )

        try:
            model, metadata = self._load_model(model_path)
        except Exception as exc:
            return JudgeResult(
                score=0.0,
                passed=False,
                metrics={},
                checks={"model_loadable": False},
                message=f"Model rejected by judge: {exc}",
            )

        data_cfg = self.config["dataset"]
        dataset = RetrievalShiftDataset(
            seed=seed,
            num_samples=int(data_cfg["num_samples"]),
            feature_dim=int(data_cfg["feature_dim"]),
            num_classes=int(data_cfg["num_classes"]),
            train_ratio=float(data_cfg["train_ratio"]),
        )

        train = dataset.get_split("train")
        val = dataset.get_split("val")
        train_metrics = self._evaluate_split(model, train.features, train.labels)
        val_metrics = self._evaluate_split(model, val.features, val.labels)

        gap = float(train_metrics["accuracy"] - val_metrics["accuracy"])
        grad_norm = self._gradient_norm(model, train.features, train.labels)

        train_dist = train_metrics["action_distribution"]
        val_dist = val_metrics["action_distribution"]
        kl_dist = float(np.sum(train_dist * (np.log(train_dist + 1e-8) - np.log(val_dist + 1e-8))))

        t = self.config["ood_thresholds"]
        checks = {
            "ood_accuracy": val_metrics["accuracy"] >= float(t["ood_accuracy_min"]),
            "entropy": val_metrics["entropy"] >= float(t["entropy_min"]),
            "gradient_norm": grad_norm <= float(t["gradient_norm_max"]),
            "generalization_gap": gap <= float(t["generalization_gap_max"]),
            "retrieval_kl": kl_dist <= float(t["retrieval_kl_max"]),
        }

        penalties = []
        penalties.append(max(0.0, float(t["ood_accuracy_min"]) - val_metrics["accuracy"]))
        penalties.append(max(0.0, float(t["entropy_min"]) - val_metrics["entropy"]))
        penalties.append(max(0.0, grad_norm - float(t["gradient_norm_max"])) / max(float(t["gradient_norm_max"]), 1e-6))
        penalties.append(max(0.0, gap - float(t["generalization_gap_max"])))
        penalties.append(max(0.0, kl_dist - float(t["retrieval_kl_max"])))

        penalty = float(np.mean(penalties))
        score = float(np.clip(1.0 - penalty, 0.0, 1.0))
        passed = all(checks.values())

        metrics = {
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_entropy": float(val_metrics["entropy"]),
            "generalization_gap": gap,
            "gradient_norm": grad_norm,
            "retrieval_distribution_kl": kl_dist,
            "checkpoint_last_grad_norm": float(metadata.get("last_grad_norm", 0.0)),
        }

        return JudgeResult(
            score=score,
            passed=passed,
            metrics=metrics,
            checks=checks,
            message="Judge evaluation complete.",
        )
