# logs/logger.py
"""
Logger — unified interface for WandB, TensorBoard, and matplotlib.

All outputs for a run are scoped under logs/<run_name>/:
    logs/<run_name>/          ← TensorBoard events
    logs/<run_name>/rewards.png

Usage:
    logger = Logger(use_wandb=True, use_tensorboard=True,
                    project="mario-ppo", run_name="20260302_Net_RLTrainer_lr2p5e-04_n_epochs4_0")
    logger.log({"reward": 42.0, "loss/policy": 0.1}, step=1000)
    logger.save_plot(episode_rewards)
    logger.close()
"""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for training scripts
import matplotlib.pyplot as plt


class Logger:
    def __init__(
        self,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        project: str = "mario-ppo",
        run_name: str = "run",
        config: Optional[dict] = None,
        resume_run_id: Optional[str] = None,
    ):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.run_name = run_name

        log_dir = os.path.join("logs", run_name)

        if use_wandb:
            import wandb
            wandb.init(project=project, name=run_name, config=config or {})
            self._wandb = wandb

        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=log_dir)

    def get_wandb_run_id(self) -> str | None:
        """Return the current run ID so it can be saved to the checkpoint."""
        if self.use_wandb:
            return self._wandb.run.id
        return None

    def log(self, metrics: dict, step: int):
        """Write metrics to WandB and/or TensorBoard."""
        if self.use_wandb:
            self._wandb.log(metrics, step=step)

        if self.use_tensorboard:
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, global_step=step)

    def save_plot(self, episode_rewards: list, path: Optional[str] = None):
        """Save a matplotlib line plot of episode rewards to logs/<run_name>/rewards.png."""
        plot_path: str = path or os.path.join("logs", self.run_name, "rewards.png")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(episode_rewards, linewidth=0.8, alpha=0.7, label="Episode reward")

        # Running mean (window=50) for readability
        if len(episode_rewards) >= 50:
            import numpy as np
            kernel = np.ones(50) / 50
            smoothed = np.convolve(episode_rewards, kernel, mode="valid")
            ax.plot(range(49, len(episode_rewards)), smoothed, linewidth=2, label="Mean-50")

        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.set_title("Mario PPO — Training rewards")
        ax.legend()
        fig.tight_layout()
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"[Logger] Saved plot → {plot_path}")

    def close(self):
        """Flush and close all backends."""
        if self.use_wandb:
            self._wandb.finish()
        if self.use_tensorboard:
            self._writer.close()
