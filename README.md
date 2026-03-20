# agent_eval_lab

![Agent Eval Lab Banner](./agent_eval_lab_banner.png)

Production-grade deterministic RL environment framework for training and evaluating LLM-style agents without external APIs.

## Overview

`agent_eval_lab` provides internal-infrastructure style components for:

- deterministic environment simulation
- execution-based deterministic judging
- PPO training with stability safeguards
- OOD evaluation under distribution shift
- reproducibility guarantees for CI workflows

## Recent Features & Improvements
- **Environments**: Added the new `label_noise` environment and fixed GAE for 1-step episodes.
- **Robustness**: Enforced deterministic PyTorch natively and secured `torch.load` operations.
- **Architecture**: Rewrote main execution to `argparse` subcommands (`train`, `evaluate`, `reproduce`) and implemented robust typed configurations (`PipelineConfig`).
- **Algorithm**: Fixed the exact KL approximation formulation in PPO for Schulman verification.
- **GPU Acceleration**: Fully integrated end-to-end device mapping (`device: "cuda"` / `"cpu"`).
- **CI/CD Validation**: Expanded full `pytest` suite and integrated GitHub Actions workflows.

The environment is a retrieval-style contextual bandit under controlled covariance and class-prior shift.

## Architecture

```text
+--------------------+       +---------------------------+
|  configs/default   |------>| seed + config hash        |
+--------------------+       +------------+--------------+
                                           |
                                           v
+--------------------+       +---------------------------+
| retrieval dataset  |------>| RetrievalShiftEnv         |
| train + shifted val|       | reward + entropy/KL guard |
+--------------------+       +------------+--------------+
                                           |
                                           v
+--------------------+       +---------------------------+
| PPOTrainer         |------>| checkpoints + metrics log |
| clipped obj + GAE  |       | stability abort checks    |
+--------------------+       +------------+--------------+
                                           |
                           +---------------+----------------+
                           v                                v
                +-----------------------+        +----------------------+
                | OOD evaluator         |        | Deterministic judge  |
                | acc/gap/retrieval KL  |        | thresholds + [0,1]   |
                +-----------------------+        +----------------------+
```

## Why deterministic judges matter

Deterministic judges remove stochastic evaluation variance and make policy quality regressions actionable in CI. Fixed seeds, stable datasets, and execution-based checks ensure failures are attributable to code/model changes rather than random rollouts.

## Distribution shift experiment

The default environment generates 1500 synthetic samples with:

- 32-dimensional features
- 5 classes
- shifted validation covariance via transformed train covariance
- shifted validation class priors (heavy class imbalance)

Training only sees train split. Validation distribution remains hidden from optimization and is used for OOD assessment.

## Reward hacking prevention strategy

Reward combines task correctness with two safeguards:

- entropy collapse penalty to discourage degenerate deterministic retrieval
- KL penalty against a fixed initial policy reference to discourage pathological policy drift

Judge checks OOD accuracy, entropy floor, gradient norm, generalization gap, and retrieval-distribution KL.

## Stability instrumentation

During PPO training, logs include:

- policy entropy
- gradient norm
- KL divergence
- reward mean and reward variance

Safety aborts trigger when:

- gradient norm spikes above 5x rolling mean
- entropy falls below 50% of initial entropy
- KL exceeds configured explosion threshold

## Run training

```bash
uv run python main.py train --config configs/default.yaml
```

Optional reproducibility verification check run:

```bash
uv run python main.py reproduce --config configs/default.yaml
```

## Run judge only

Use Python shell or script:

```python
import yaml
from core.config_schema import PipelineConfig
from environments.retrieval_shift.judge import RetrievalShiftJudge

cfg = PipelineConfig.from_dict(yaml.safe_load(open("configs/default.yaml", "r", encoding="utf-8")))
judge = RetrievalShiftJudge(cfg)
print(judge.evaluate("outputs/default_run/policy.pt"))
```

## Example output metrics

```json
{
  "ood_metrics": {
    "train_accuracy": 0.67,
    "val_accuracy": 0.52,
    "generalization_gap": 0.15,
    "retrieval_distribution_kl": 0.11
  },
  "judge": {
    "score": 0.92,
    "passed": true
  }
}
```

## Reproducibility guarantees

- global seed for Python, NumPy, and PyTorch
- deterministic PyTorch mode enabled
- config hash logged with each iteration
- optional dual-run reproducibility check via config flag
