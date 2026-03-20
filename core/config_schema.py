"""Typed configuration schema for deterministic RL pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DatasetConfig:
    num_samples: int
    feature_dim: int
    num_classes: int
    train_ratio: float

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> DatasetConfig:
        try:
            return cls(
                num_samples=int(raw["num_samples"]),
                feature_dim=int(raw["feature_dim"]),
                num_classes=int(raw["num_classes"]),
                train_ratio=float(raw["train_ratio"]),
            )
        except KeyError as e:
            raise ValueError(f"Missing required key in DatasetConfig: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid type in DatasetConfig: {e}")


@dataclass
class EnvConfig:
    name: str
    entropy_penalty_coef: float
    kl_penalty_coef: float
    kl_threshold: float
    top_k: int
    noise_rate: float = 0.0

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> EnvConfig:
        try:
            return cls(
                name=str(raw.get("name", "retrieval_shift")),
                entropy_penalty_coef=float(raw["entropy_penalty_coef"]),
                kl_penalty_coef=float(raw["kl_penalty_coef"]),
                kl_threshold=float(raw["kl_threshold"]),
                top_k=int(raw["top_k"]),
                noise_rate=float(raw.get("noise_rate", 0.0)),
            )
        except KeyError as e:
            raise ValueError(f"Missing required key in EnvConfig: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid type in EnvConfig: {e}")


@dataclass
class TrainingConfig:
    learning_rate: float
    clip_epsilon: float
    entropy_coef: float
    value_coef: float
    gamma: float
    gae_lambda: float
    max_grad_norm: float
    train_iterations: int
    steps_per_iter: int
    update_epochs: int
    minibatch_size: int
    kl_abort_threshold: float
    hidden_dim: int
    output_dir: str
    device: str = "cpu"

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> TrainingConfig:
        try:
            return cls(
                learning_rate=float(raw["learning_rate"]),
                clip_epsilon=float(raw["clip_epsilon"]),
                entropy_coef=float(raw["entropy_coef"]),
                value_coef=float(raw["value_coef"]),
                gamma=float(raw["gamma"]),
                gae_lambda=float(raw["gae_lambda"]),
                max_grad_norm=float(raw["max_grad_norm"]),
                train_iterations=int(raw["train_iterations"]),
                steps_per_iter=int(raw["steps_per_iter"]),
                update_epochs=int(raw["update_epochs"]),
                minibatch_size=int(raw["minibatch_size"]),
                kl_abort_threshold=float(raw["kl_abort_threshold"]),
                hidden_dim=int(raw.get("hidden_dim", 128)),
                output_dir=str(raw["output_dir"]),
                device=str(raw.get("device", "cpu")),
            )
        except KeyError as e:
            raise ValueError(f"Missing required key in TrainingConfig: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid type in TrainingConfig: {e}")


@dataclass
class OODThresholds:
    ood_accuracy_min: float
    entropy_min: float
    gradient_norm_max: float
    generalization_gap_max: float
    retrieval_kl_max: float

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> OODThresholds:
        try:
            return cls(
                ood_accuracy_min=float(raw["ood_accuracy_min"]),
                entropy_min=float(raw["entropy_min"]),
                gradient_norm_max=float(raw["gradient_norm_max"]),
                generalization_gap_max=float(raw["generalization_gap_max"]),
                retrieval_kl_max=float(raw["retrieval_kl_max"]),
            )
        except KeyError as e:
            raise ValueError(f"Missing required key in OODThresholds: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid type in OODThresholds: {e}")


@dataclass
class PipelineConfig:
    seed: int
    logging_level: str
    dataset: DatasetConfig
    env: EnvConfig
    training: TrainingConfig
    ood_thresholds: OODThresholds
    dual_run_test: bool

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> PipelineConfig:
        try:
            return cls(
                seed=int(raw["seed"]),
                logging_level=str(raw.get("logging", {}).get("level", "INFO")),
                dataset=DatasetConfig.from_dict(raw["dataset"]),
                env=EnvConfig.from_dict(raw["env"]),
                training=TrainingConfig.from_dict(raw["training"]),
                ood_thresholds=OODThresholds.from_dict(raw["ood_thresholds"]),
                dual_run_test=bool(raw.get("reproducibility", {}).get("dual_run_test", False)),
            )
        except KeyError as e:
            raise ValueError(f"Missing required key in PipelineConfig: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid type in PipelineConfig: {e}")
