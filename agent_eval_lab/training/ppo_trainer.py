"""Minimal deterministic PPO trainer with stability instrumentation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from core.seed import config_hash
from evaluation.stability_metrics import gradient_spike, rolling_mean
from training.advantage import compute_gae
from training.logging_utils import MetricsLogger

LOGGER = logging.getLogger(__name__)


class PolicyValueNet(nn.Module):
    """Shared-backbone policy and value network."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


@dataclass
class PPOConfig:
    """Trainer hyperparameters."""

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
    output_dir: str
    seed: int


class PPOTrainer:
    """Deterministic PPO training loop for one-step contextual bandit tasks."""

    def __init__(self, env: Any, config: Dict[str, Any]) -> None:
        train_cfg = config["training"]
        self.cfg = PPOConfig(
            learning_rate=float(train_cfg["learning_rate"]),
            clip_epsilon=float(train_cfg["clip_epsilon"]),
            entropy_coef=float(train_cfg["entropy_coef"]),
            value_coef=float(train_cfg["value_coef"]),
            gamma=float(train_cfg["gamma"]),
            gae_lambda=float(train_cfg["gae_lambda"]),
            max_grad_norm=float(train_cfg["max_grad_norm"]),
            train_iterations=int(train_cfg["train_iterations"]),
            steps_per_iter=int(train_cfg["steps_per_iter"]),
            update_epochs=int(train_cfg["update_epochs"]),
            minibatch_size=int(train_cfg["minibatch_size"]),
            kl_abort_threshold=float(train_cfg["kl_abort_threshold"]),
            output_dir=str(train_cfg["output_dir"]),
            seed=int(config["seed"]),
        )

        self.env = env
        self.device = torch.device("cpu")
        self.model = PolicyValueNet(env.feature_dim, env.num_classes, hidden_dim=int(train_cfg["hidden_dim"]))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        self.metrics_logger = MetricsLogger(os.path.join(self.cfg.output_dir, "training_metrics.jsonl"))
        self.grad_history: List[float] = []

        self.config_hash = config_hash(config)
        LOGGER.info("Config hash: %s", self.config_hash)

    @staticmethod
    def _to_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array.astype(np.float32))

    def _collect_rollout(self) -> Dict[str, np.ndarray]:
        obs_buf: List[np.ndarray] = []
        act_buf: List[int] = []
        logp_buf: List[float] = []
        rew_buf: List[float] = []
        val_buf: List[float] = []
        done_buf: List[float] = []
        ent_buf: List[float] = []

        for _ in range(self.cfg.steps_per_iter):
            obs = self.env.reset()
            obs_t = self._to_tensor(obs).unsqueeze(0)

            with torch.no_grad():
                logits, value = self.model(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                entropy = float(dist.entropy().item())

            _, reward, done, _ = self.env.step({"action": int(action.item()), "probs": probs})

            obs_buf.append(obs)
            act_buf.append(int(action.item()))
            logp_buf.append(float(logp.item()))
            rew_buf.append(float(reward))
            val_buf.append(float(value.item()))
            done_buf.append(float(done))
            ent_buf.append(entropy)

        rewards = np.asarray(rew_buf, dtype=np.float32)
        values = np.asarray(val_buf, dtype=np.float32)
        dones = np.asarray(done_buf, dtype=np.float32)
        advantages, returns = compute_gae(rewards, values, dones, self.cfg.gamma, self.cfg.gae_lambda)

        return {
            "obs": np.asarray(obs_buf, dtype=np.float32),
            "actions": np.asarray(act_buf, dtype=np.int64),
            "logp": np.asarray(logp_buf, dtype=np.float32),
            "rewards": rewards,
            "values": values,
            "advantages": advantages,
            "returns": returns,
            "entropy": np.asarray(ent_buf, dtype=np.float32),
        }

    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += float(torch.sum(p.grad.detach() ** 2).item())
        return float(np.sqrt(total))

    def _ppo_update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = self._to_tensor(batch["obs"])
        actions = torch.from_numpy(batch["actions"])
        old_logp = torch.from_numpy(batch["logp"])
        advantages = torch.from_numpy(batch["advantages"])
        returns = torch.from_numpy(batch["returns"])

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)

        last_grad_norm = 0.0
        last_kl = 0.0
        last_entropy = 0.0

        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.cfg.minibatch_size):
                mb_idx = idx[start : start + self.cfg.minibatch_size]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                logits, values = self.model(mb_obs)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                last_grad_norm = self._grad_norm()
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = torch.mean(mb_old_logp - new_logp).item()
                last_kl = float(approx_kl)
                last_entropy = float(entropy.item())

        reward_mean = float(np.mean(batch["rewards"]))
        reward_var = float(np.var(batch["rewards"]))
        return {
            "reward_mean": reward_mean,
            "reward_var": reward_var,
            "entropy": last_entropy,
            "grad_norm": last_grad_norm,
            "kl": last_kl,
        }

    def train(self) -> Dict[str, float]:
        initial_entropy = None
        aborted = False
        abort_reason = ""
        latest_metrics: Dict[str, float] = {}

        for itr in range(1, self.cfg.train_iterations + 1):
            batch = self._collect_rollout()
            metrics = self._ppo_update(batch)

            if initial_entropy is None:
                initial_entropy = metrics["entropy"]

            self.grad_history.append(metrics["grad_norm"])

            if gradient_spike(metrics["grad_norm"], self.grad_history[:-1], factor=5.0, window=20):
                aborted = True
                abort_reason = "Gradient norm spike detected."

            if initial_entropy is not None and metrics["entropy"] < 0.5 * initial_entropy:
                aborted = True
                abort_reason = "Entropy collapse detected."

            if metrics["kl"] > self.cfg.kl_abort_threshold:
                aborted = True
                abort_reason = "KL divergence explosion detected."

            payload = {
                "iteration": itr,
                "reward_mean": metrics["reward_mean"],
                "reward_var": metrics["reward_var"],
                "policy_entropy": metrics["entropy"],
                "gradient_norm": metrics["grad_norm"],
                "kl_divergence": metrics["kl"],
                "rolling_grad_mean": rolling_mean(self.grad_history, 20),
                "config_hash": self.config_hash,
                "aborted": aborted,
            }
            self.metrics_logger.log(payload)
            LOGGER.info(
                "iter=%d reward_mean=%.4f entropy=%.4f grad=%.4f kl=%.4f",
                itr,
                metrics["reward_mean"],
                metrics["entropy"],
                metrics["grad_norm"],
                metrics["kl"],
            )

            latest_metrics = payload
            if aborted:
                LOGGER.error("Training aborted: %s", abort_reason)
                latest_metrics["abort_reason"] = abort_reason
                break

        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "obs_dim": self.env.feature_dim,
            "num_actions": self.env.num_classes,
            "hidden_dim": self.model.hidden_dim,
            "metadata": {
                "model_type": "mlp_policy",
                "seed": self.cfg.seed,
                "last_grad_norm": latest_metrics.get("gradient_norm", 0.0),
                "initial_entropy": initial_entropy if initial_entropy is not None else 0.0,
                "config_hash": self.config_hash,
            },
        }
        ckpt_path = os.path.join(self.cfg.output_dir, "policy.pt")
        torch.save(ckpt, ckpt_path)
        latest_metrics["checkpoint_path"] = ckpt_path
        return latest_metrics
