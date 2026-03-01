# trainer/base_trainer.py
"""
Trainer — abstract base class.

Provides:
  save_checkpoint(path)
  load_checkpoint(path)
"""

import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.optim import Optimizer

from logs.logger import Logger


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        optimizer: Optimizer,
        config: dict,
        logger: Logger,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config
        self.logger = logger

    @abstractmethod
    def train_loop(self, batch) -> torch.Tensor:
        """
        Perform one gradient update on a mini-batch.

        Args:
            batch: tuple of tensors returned by RolloutBuffer.get_batches()

        Returns:
            Scalar loss tensor (after .backward() has been called).
        """

    @abstractmethod
    def test_loop(self, env, n_episodes: int) -> float:
        """
        Evaluate the current policy for n_episodes, return mean total reward.
        """

    def save_checkpoint(self, path: str):
        """Save model + optimizer state to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "global_step": getattr(self, "_global_step", 0),
                "wandb_run_id": self.logger.get_wandb_run_id(),
            },
            path,
        )
        print(f"[Trainer] Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        """Load model + optimizer state from disk."""
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"[Trainer] Checkpoint loaded ← {path}")
