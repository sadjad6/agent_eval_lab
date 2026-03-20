# Codebase Index: agent_eval_lab

## Architecture Overview
`agent_eval_lab` is a production-grade deterministic RL environment framework designed for training and evaluating LLM-style agents without external APIs. It operates as a retrieval-style contextual bandit that systematically introduces controlled covariance and class-prior shifts.

## Directory Structure and Core Modules

### Root Files
- `main.py`
  - **Purpose**: Entry point for the RL training and evaluation pipeline. 
  - **Key Functions**: `run_pipeline`, `reproducibility_check`.
- `README.md`: High-level overview, architecture diagram, and usage instructions.
- `requirements.txt`: Python package dependencies.
- `Dockerfile`: Containerization instructions for reproducible runs.

### `core/`
Foundational interfaces and utilities.
- `env_interface.py`: Defines `EnvInterface`, the abstract base class for RL environments.
- `judge_interface.py`: Defines `JudgeInterface`, the abstract base class for deterministic judges.
- `seed.py`: Utilities for global seeding (`set_global_seed`) and configuration hashing (`config_hash`) to ensure absolute reproducibility.

### `environments/`
Contains the specific RL environment implementations.
- `retrieval_shift/`
  - `env.py`: Implements `RetrievalShiftEnv`. Handles step execution, custom reward formulation, and calculates stability penalties (entropy bounds and KL-to-initial divergence).
  - `dataset.py`: Implements `RetrievalShiftDataset`. Generates synthetic, shifted datasets for train and validation phases.
  - `judge.py`: Implements `RetrievalShiftJudge`. Provides execution-based deterministic judging with strict pass/fail thresholds.

### `training/`
PPO reinforcement learning loop and optimizations.
- `ppo_trainer.py`: Implements `PPOTrainer` and `PolicyValueNet`. Executes the PPO training loop with minibatch updates, gradient clipping, advantage scaling, and stability abort checks (gradient spikes, entropy collapse, KL explosion).
- `advantage.py`: Implements Generalized Advantage Estimation (GAE).
- `logging_utils.py`: Provides structured logging (`MetricsLogger`).

### `evaluation/`
Metrics and Out-Of-Distribution (OOD) evaluation.
- `ood_eval.py`: Includes `evaluate_ood` to test policies on shifted validation datasets.
- `stability_metrics.py`: Utility functions like `gradient_spike` and `rolling_mean`.

### `configs/`
- YAML configuration files controlling hyperparameters, environment settings, and reproducibility flags (e.g., `default.yaml`).

### `tests/`
- Unit testing suite checking pipeline steps, environment behavior, and deterministic properties.

## Execution Flows
1. **Training Run**: 
   `main.py` -> Loads Config -> Initiates `RetrievalShiftEnv` & `PPOTrainer` -> Outputs Checkpoint.
2. **OOD Evaluation & Judging**: 
   `main.py` -> Calls `evaluate_ood` -> Passes results to `RetrievalShiftJudge` -> Returns final score/metrics.
3. **Reproducibility Test**: 
   `main.py` -> Runs `reproducibility_check(config)` -> Executes pipeline twice -> Compares outputs ensuring exact identical distributions.
