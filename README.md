# agent_eval_lab

Experimental framework for training and evaluating small policy networks in synthetic, one-step contextual-bandit environments. It uses a custom PyTorch PPO-style trainer and threshold-based checkpoint judges without external model APIs; it does not train or evaluate an LLM.

## Overview

`agent_eval_lab` provides internal-infrastructure style components for:

- seeded synthetic environment simulation
- threshold-based checkpoint evaluation
- PPO-style training with stability instrumentation
- OOD evaluation under controlled synthetic distribution shift
- repeat-run comparison of four evaluation metrics

## Recent Features & Improvements
- **Environments**: Includes `retrieval_shift` and a partially integrated `label_noise` training branch. The shared OOD evaluator and standalone `evaluate` command still select retrieval-shift components.
- **Robustness**: Seeds Python, NumPy and PyTorch; deterministic algorithms use `warn_only=True`. Checkpoint loading uses `weights_only=True`.
- **Architecture**: Rewrote main execution to `argparse` subcommands (`train`, `evaluate`, `reproduce`) and implemented robust typed configurations (`PipelineConfig`).
- **Algorithm**: Implements clipped PPO-style updates, gradient clipping, entropy/value terms and KL instrumentation.
- **Device Mapping**: Configurable CPU/CUDA device mapping is present. The default configuration and CI use CPU; GPU validation is not established.
- **CI Configuration**: A pytest suite and GitHub Actions test/train/reproduce steps are included. The current default-branch test job is failing; a workflow file is not evidence of passing CI.

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

## Why seeded evaluation matters

Fixed seeds, synthetic datasets and threshold checks help compare checkpoint behavior. These controls reduce sources of evaluation variance, but do not guarantee identical artifacts across hardware or dependency versions. The judges execute policy networks on synthetic data; they do not judge LLM responses or tool use.

## Distribution shift experiment

The default environment generates 1500 synthetic samples with:

- 32-dimensional features
- 5 classes
- shifted validation covariance via transformed train covariance
- shifted validation class priors (heavy class imbalance)

Training only sees train split. Validation distribution remains hidden from optimization and is used for OOD assessment.

## Experimental reward design

The reward uses top-k policy-probability rankings rather than the sampled action. It combines this synthetic correctness signal with two penalties; their presence does not establish prevention of reward hacking:

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
python -m pip install -r requirements.txt
python main.py train --config configs/default.yaml
```

Optional reproducibility verification check run:

```bash
python main.py reproduce --config configs/default.yaml
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

## Output artifacts

Training writes checkpoints and JSONL metrics under `outputs/default_run/`. Numerical performance should be reported with the corresponding run artifacts and configuration; no benchmark scores are claimed here.

## Reproducibility controls and limits

- Global seeds for Python, NumPy and PyTorch
- PyTorch deterministic algorithms enabled with `warn_only=True`
- Config hash recorded with the run
- Optional dual-run check compares four OOD metrics, not every output or hardware configuration
- Default retrieval-shift evaluation is synthetic; label-noise evaluation remains only partially integrated
