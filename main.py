"""Entry point for deterministic RL training and evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict

import yaml

from core.config_schema import PipelineConfig
from core.seed import config_hash, set_global_seed
from environments.retrieval_shift.dataset import RetrievalShiftDataset
from environments.retrieval_shift.env import RetrievalShiftEnv
from environments.retrieval_shift.judge import RetrievalShiftJudge
from evaluation.ood_eval import evaluate_ood
from training.logging_utils import configure_logging
from training.ppo_trainer import PPOTrainer


def load_config(path: str) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig.from_dict(raw)


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    set_global_seed(config.seed)
    configure_logging(config.logging_level)

    env_name = getattr(config.env, "name", "retrieval_shift")
    if env_name == "retrieval_shift":
        dataset = RetrievalShiftDataset(
            seed=config.seed,
            num_samples=config.dataset.num_samples,
            feature_dim=config.dataset.feature_dim,
            num_classes=config.dataset.num_classes,
            train_ratio=config.dataset.train_ratio,
        )
        env = RetrievalShiftEnv(
            dataset=dataset,
            split="train",
            seed=config.seed,
            entropy_penalty_coef=config.env.entropy_penalty_coef,
            kl_penalty_coef=config.env.kl_penalty_coef,
            kl_threshold=config.env.kl_threshold,
            top_k=config.env.top_k,
        )
    elif env_name == "label_noise":
        from environments.label_noise.dataset import LabelNoiseDataset
        from environments.label_noise.env import LabelNoiseEnv
        dataset = LabelNoiseDataset(
            seed=config.seed,
            num_samples=config.dataset.num_samples,
            feature_dim=config.dataset.feature_dim,
            num_classes=config.dataset.num_classes,
            train_ratio=config.dataset.train_ratio,
            noise_rate=config.env.noise_rate,
        )
        env = LabelNoiseEnv(
            dataset=dataset,
            split="train",
            seed=config.seed,
            entropy_penalty_coef=config.env.entropy_penalty_coef,
            kl_penalty_coef=config.env.kl_penalty_coef,
            kl_threshold=config.env.kl_threshold,
            top_k=config.env.top_k,
        )
    else:
        raise ValueError(f"Unknown env name: {env_name}")

    trainer = PPOTrainer(env, config)
    train_metrics = trainer.train()
    model_path = train_metrics["checkpoint_path"]

    ood_metrics = evaluate_ood(model_path, config, device=config.training.device)
    
    env_name = getattr(config.env, "name", "retrieval_shift")
    if env_name == "retrieval_shift":
        judge = RetrievalShiftJudge(config)
    elif env_name == "label_noise":
        from environments.label_noise.judge import LabelNoiseJudge
        judge = LabelNoiseJudge(config)
    else:
        raise ValueError(f"Unknown env name: {env_name}")

    judge_result = judge.evaluate(model_path, device=config.training.device)

    return {
        "config_hash": config_hash(asdict(config)),
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


def reproducibility_check(config: PipelineConfig) -> Dict[str, Any]:
    """Run two identical pipelines and compare deterministic outputs."""
    run_a_cfg = copy.deepcopy(config)
    run_a_cfg.training.output_dir = os.path.join("outputs", "repro_run_a")

    run_b_cfg = copy.deepcopy(config)
    run_b_cfg.training.output_dir = os.path.join("outputs", "repro_run_b")

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
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    train_parser = subparsers.add_parser("train", help="Run the full training pipeline")
    train_parser.add_argument("--config", required=True, help="Path to YAML config file")
    train_parser.add_argument(
        "--compare-no-entropy",
        action="store_true",
        help="Run a second comparison experiment with entropy disabled.",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model on OOD sets")
    eval_parser.add_argument("--config", required=True, help="Path to YAML config file")
    eval_parser.add_argument("--model-path", required=True, help="Path to policy.pt checkpoint")

    repro_parser = subparsers.add_parser("reproduce", help="Run dual-pipeline reproducibility check")
    repro_parser.add_argument("--config", required=True, help="Path to YAML config file")

    if len(sys.argv) > 1 and sys.argv[1] not in {"train", "evaluate", "reproduce", "-h", "--help"}:
        parser.add_argument("--config", required=True, help="Path to YAML config file")
        parser.add_argument("--compare-no-entropy", action="store_true")
        args = parser.parse_args()
        args.command = "train"
    else:
        args = parser.parse_args()
        if args.command is None:
            parser.print_help()
            sys.exit(1)

    config = load_config(args.config)

    if args.command == "train":
        result = run_pipeline(config)
        if getattr(config, "dual_run_test", False):
            result["reproducibility"] = reproducibility_check(config)
        print(json.dumps(result, indent=2, sort_keys=True))

        if args.compare_no_entropy:
            no_ent_cfg = copy.deepcopy(config)
            no_ent_cfg.training.entropy_coef = 0.0
            no_ent_cfg.training.output_dir = os.path.join("outputs", "no_entropy_run")
            no_ent_result = run_pipeline(no_ent_cfg)
            print(json.dumps({"no_entropy_run": no_ent_result}, indent=2, sort_keys=True))
            
    elif args.command == "evaluate":
        ood_metrics = evaluate_ood(args.model_path, config, config.training.device)
        judge = RetrievalShiftJudge(config)
        judge_res = judge.evaluate(args.model_path, config.training.device)
        out = {
            "ood_metrics": ood_metrics,
            "judge": {"score": judge_res.score, "passed": judge_res.passed}
        }
        print(json.dumps(out, indent=2))
        
    elif args.command == "reproduce":
        result = reproducibility_check(config)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
