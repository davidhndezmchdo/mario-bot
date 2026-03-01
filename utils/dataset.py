"""
RolloutBuffer — stores one rollout of (N env steps × 1 env) before each PPO update.

WHY GAE (Generalized Advantage Estimation):
  Advantage A(s,a) = Q(s,a) - V(s) measures how much better action a was
  compared to the average. A naive estimate uses TD residuals (low variance,
  high bias) or Monte-Carlo returns (low bias, high variance). GAE-λ
  interpolates between the two:

      δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)          (TD residual)
      Aₜ = δₜ + (γλ)δₜ₊₁ + (γλ)²δₜ₊₂ + …

  λ=0.95, γ=0.99 is the standard setting. Higher λ → more variance, less bias.

Layout:
  self.obs       : list of np.ndarray (240, 256, 4)
  self.actions   : list of int
  self.rewards   : list of float
  self.dones     : list of bool
  self.values    : list of float  (V(sₜ) from critic)
  self.log_probs : list of float  (log π(aₜ|sₜ))

After compute_advantages():
  self.advantages : np.ndarray (N,)
  self.returns    : np.ndarray (N,)   used as critic targets
"""

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self):
        self.obs: list = []
        self.actions: list = []
        self.rewards: list = []
        self.dones: list = []
        self.values: list = []
        self.log_probs: list = []

        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        """Append one time-step of experience."""
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute GAE-λ advantages and discounted returns in-place.

        Args:
            last_value: V(s_{T+1}) from the critic — bootstrap for the last step.
            gamma:      discount factor
            gae_lambda: GAE smoothing parameter (0=TD, 1=MC)
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            next_non_terminal = 1.0 - (dones[t])
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        self.advantages = advantages
        self.returns = advantages + values  # V-targets for critic

    def get_batches(self, batch_size: int, device: str = "cpu"):
        """
        Yield shuffled mini-batches as tensors ready for the PPO update.

        Yields:
            obs_b:      (B, 4, H, W) float32 in [0, 1]
            act_b:      (B,) int64
            lp_b:       (B,) float32  — old log_probs
            adv_b:      (B,) float32  — normalized advantages
            returns_b:  (B,) float32
        """
        n = len(self.obs)
        indices = np.random.permutation(n)

        # Normalise advantages over the whole rollout (reduces gradient variance)
        adv = self.advantages.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_arr = np.array(self.obs, dtype=np.float32) / 255.0  # (N, H, W, 4)
        # Reorder to (N, 4, H, W) — PyTorch expects CHW
        obs_arr = np.transpose(obs_arr, (0, 3, 1, 2))
        act_arr = np.array(self.actions, dtype=np.int64)
        lp_arr = np.array(self.log_probs, dtype=np.float32)
        ret_arr = self.returns.astype(np.float32)

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            yield (
                torch.tensor(obs_arr[idx], device=device),
                torch.tensor(act_arr[idx], device=device),
                torch.tensor(lp_arr[idx], device=device),
                torch.tensor(adv[idx], device=device),
                torch.tensor(ret_arr[idx], device=device),
            )

    def clear(self):
        """Reset buffer after each PPO update."""
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.rewards)
