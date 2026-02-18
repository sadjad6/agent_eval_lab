"""Entry point for deterministic RL training and evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict

import yaml

from core.seed import config_hash, set_global_seed
from environments.retrieval_shift.dataset import RetrievalShiftDataset
from environments.retrieval_shift.env import RetrievalShiftEnv
from environments.retrieval_shift.judge import RetrievalShiftJudge
from evaluation.ood_eval import evaluate_ood
from training.logging_utils import configure_logging
from training.ppo_trainer import PPOTrainer


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    set_global_seed(int(config["seed"]))
    configure_logging(config.get("logging", {}).get("level", "INFO"))

    dcfg = config["dataset"]
    dataset = RetrievalShiftDataset(
        seed=int(config["seed"]),
        num_samples=int(dcfg["num_samples"]),
        feature_dim=int(dcfg["feature_dim"]),
        num_classes=int(dcfg["num_classes"]),
        train_ratio=float(dcfg["train_ratio"]),
    )

    ecfg = config["env"]
    env = RetrievalShiftEnv(
        dataset=dataset,
        split="train",
        seed=int(config["seed"]),
        entropy_penalty_coef=float(ecfg["entropy_penalty_coef"]),
        kl_penalty_coef=float(ecfg["kl_penalty_coef"]),
        kl_threshold=float(ecfg["kl_threshold"]),
        top_k=int(ecfg["top_k"]),
    )

    trainer = PPOTrainer(env, config)
    train_metrics = trainer.train()
    model_path = train_metrics["checkpoint_path"]

    ood_metrics = evaluate_ood(model_path, config)
    judge = RetrievalShiftJudge(config)
    judge_result = judge.evaluate(model_path)

    return {
        "config_hash": config_hash(config),
        "dataset_shift": dataset.shift_metrics(),
        "train_metrics": train_metrics,
        "ood_metrics": ood_metrics,
        "judge": {
            "score": judge_result.score,
            "passed": judge_result.passed,
            "checks": judge_result.checks,
            "metrics": judge_result.metrics,
            "message": judge_result.message,
        },
    }


def reproducibility_check(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run two identical pipelines and compare deterministic outputs."""
    run_a_cfg = copy.deepcopy(config)
    run_a_cfg["training"]["output_dir"] = os.path.join("outputs", "repro_run_a")

    run_b_cfg = copy.deepcopy(config)
    run_b_cfg["training"]["output_dir"] = os.path.join("outputs", "repro_run_b")

    a = run_pipeline(run_a_cfg)
    b = run_pipeline(run_b_cfg)

    keys = ["train_accuracy", "val_accuracy", "generalization_gap", "retrieval_distribution_kl"]
    deltas = {
        key: abs(float(a["ood_metrics"][key]) - float(b["ood_metrics"][key]))
        for key in keys
    }
    passed = all(delta < 1e-9 for delta in deltas.values())
    return {"passed": passed, "metric_abs_delta": deltas}


def main() -> None:
    parser = argparse.ArgumentParser(description="agent_eval_lab deterministic RL pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--compare-no-entropy",
        action="store_true",
        help="Run a second comparison experiment with entropy regularization disabled.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_pipeline(config)

    if config.get("reproducibility", {}).get("dual_run_test", False):
        result["reproducibility"] = reproducibility_check(config)

    print(json.dumps(result, indent=2, sort_keys=True))

    if args.compare_no_entropy:
        no_ent_cfg = copy.deepcopy(config)
        no_ent_cfg["training"]["entropy_coef"] = 0.0
        no_ent_cfg["training"]["output_dir"] = os.path.join("outputs", "no_entropy_run")
        no_ent_result = run_pipeline(no_ent_cfg)
        print(json.dumps({"no_entropy_run": no_ent_result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
