"""
PPO Clipped Surrogate Loss.

Total loss = policy_loss + 0.5·value_loss + 0.01·entropy_loss

WHY clipping (PPO's core innovation):
  Without clipping, large policy updates can collapse performance because the
  new policy is evaluated on data collected by the old policy. The clipped
  surrogate objective limits the ratio r = π_new/π_old to [1-ε, 1+ε],
  preventing over-optimistic gradient steps.

  L_CLIP = E[ min( r·A, clip(r, 1-ε, 1+ε)·A ) ]

WHY value loss coefficient 0.5:
  Keeps critic gradient roughly on the same scale as the actor gradient.

WHY entropy bonus (coefficient 0.01):
  Penalises low-entropy (overconfident) policies during early training.
  Encourages exploration — without it the policy can prematurely collapse
  to always picking the same action.
"""

import torch
import torch.nn.functional as F


class PPOLoss:
    def __init__(
        self,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def compute(
        self,
        new_log_probs: torch.Tensor,  # (B,)
        old_log_probs: torch.Tensor,  # (B,)  — from rollout, no grad
        advantages: torch.Tensor,     # (B,)  — already normalised
        values: torch.Tensor,         # (B,)  — current critic estimates
        returns: torch.Tensor,        # (B,)  — GAE returns (critic targets)
        entropy: torch.Tensor,        # (B,)  — distribution entropy
    ):
        """
        Compute total PPO loss and return a dict of individual components for logging.

        Returns:
            total_loss: scalar tensor (call .backward() on this)
            info:       dict with float values for policy, value, entropy losses
        """
        # --- Policy loss ---
        ratio = torch.exp(new_log_probs - old_log_probs)   # π_new / π_old
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

        # Take the pessimistic (min) of clipped and unclipped objectives
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # --- Value loss ---
        value_loss = F.mse_loss(values, returns)

        # --- Entropy bonus (negative because we maximise entropy) ---
        entropy_loss = -entropy.mean()

        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        info = {
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "loss/entropy": entropy_loss.item(),
            "loss/total": total_loss.item(),
        }
        return total_loss, info
