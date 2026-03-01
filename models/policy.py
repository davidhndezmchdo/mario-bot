# models/policy.py
"""
Net — shared convolutional backbone with two output heads.

Architecture:
  Input: (B, 4, 240, 256)  — 4 stacked grayscale frames, CHW layout

  Shared CNN backbone:
    Conv2d(4→32,  kernel=8, stride=4) + ReLU   → (B, 32, 59, 63)
    Conv2d(32→64, kernel=4, stride=2) + ReLU   → (B, 64, 28, 30)
    Conv2d(64→64, kernel=3, stride=1) + ReLU   → (B, 64, 26, 28)
    Flatten                                     → (B, 46592)
    Linear(46592 → 512)                + ReLU

  Actor head:  Linear(512 → n_actions)  → logits for Categorical dist
  Critic head: Linear(512 → 1)          → scalar value estimate

WHY shared backbone:
  Both policy and value function need similar low-level visual features
  (edges, motion, game objects). Sharing reduces parameter count and
  lets gradient signals from both heads jointly improve the features.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Net(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size dynamically — avoids hard-coding magic numbers
        dummy = torch.zeros(1, 4, 240, 256)
        flat_size = self.backbone(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(512, n_actions)
        self.critic_head = nn.Linear(512, 1)

    def _shared_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Run obs through backbone + linear layer. obs: (B, 4, H, W), float in [0,1]."""
        return self.linear(self.backbone(obs))

    def forward(self, obs: torch.Tensor):
        """
        Returns raw logits and value estimate.

        Args:
            obs: (B, 4, H, W) float tensor in [0, 1]

        Returns:
            logits: (B, n_actions)
            value:  (B, 1)
        """
        features = self._shared_features(obs)
        return self.actor_head(features), self.critic_head(features)

    def get_action(self, obs: torch.Tensor):
        """
        Sample an action during rollout collection.

        Args:
            obs: (B, 4, H, W) float tensor

        Returns:
            action:   (B,) int64 — sampled action index
            log_prob: (B,) float — log probability of the sampled action
            value:    (B,) float — critic estimate
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Re-evaluate actions using the *current* policy — called during PPO update.

        Args:
            obs:     (B, 4, H, W) float tensor
            actions: (B,) int64

        Returns:
            log_prob: (B,) — log prob under current policy
            value:    (B,) — current value estimate
            entropy:  (B,) — distribution entropy (used for entropy bonus)
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value.squeeze(-1), entropy
