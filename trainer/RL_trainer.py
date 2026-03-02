"""
RLTrainer — PPO training loop.

Inherits:  Trainer (base_trainer.py)
Overrides: train_loop(), test_loop()
Adds:      train(), collect_rollout()

Training flow:
  while global_step < total_timesteps:
    1. collect_rollout(n_steps=512)
       → fill RolloutBuffer with (obs, action, reward, done, value, log_prob)
    2. buffer.compute_advantages(last_value, gamma, gae_lambda)
    3. for epoch in range(n_epochs=10):
         for batch in buffer.get_batches(batch_size=64):
           train_loop(batch)  → one Adam step
    4. buffer.clear()
    5. logger.log(metrics)
    6. if checkpoint_freq: save_checkpoint()
"""

import os
from typing import Optional

import numpy as np
import torch

from trainer.base_trainer import Trainer
from utils.dataset import RolloutBuffer


class RLTrainer(Trainer):
    def train_loop(self, batch) -> torch.Tensor:
        """
        One PPO mini-batch update.

        Args:
            batch: (obs, actions, old_log_probs, advantages, returns)

        Returns:
            total_loss tensor (already .backward()'d)
        """
        obs, actions, old_log_probs, advantages, returns = batch

        new_log_probs, values, entropy = self.model.evaluate_actions(obs, actions)

        total_loss, info = self.loss_fn.compute(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            values=values,
            returns=returns,
            entropy=entropy,
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping — prevents exploding gradients common in RL
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Stash latest loss info for logging in train()
        self._last_loss_info = info
        return total_loss

    def test_loop(self, env, n_episodes: int = 5) -> float:
        """
        Run greedy policy (argmax actions) for n_episodes.
        Returns mean total episode reward.
        """
        device = next(self.model.parameters()).device
        self.model.eval()
        episode_rewards = []

        with torch.no_grad():
            for _ in range(n_episodes):
                obs = env.reset()
                total_reward = 0.0
                done = False
                while not done:
                    obs_t = self._preprocess_obs(obs, device)
                    logits, _ = self.model(obs_t)
                    action = logits.argmax(dim=-1).item()
                    obs, reward, dones, _ = env.step([action])
                    total_reward += float(reward[0])
                    done = bool(dones[0])
                episode_rewards.append(total_reward)

        self.model.train()
        mean_reward = float(np.mean(episode_rewards))
        print(f"[Test] Mean reward over {n_episodes} episodes: {mean_reward:.2f}")
        return mean_reward

    # ------------------------------------------------------------------
    # Core training entry point
    # ------------------------------------------------------------------

    def train(self, env, total_timesteps: int, run_name: str = "run", augmenter=None):
        """
        Full PPO training loop.

        Args:
            env:             Vectorized environment from make_env()
            total_timesteps: Stop after this many env steps
            run_name:        Run identifier — weights and logs are namespaced under it
            augmenter:       Optional ObsAugmenter (or None)
        """
        device = next(self.model.parameters()).device
        buffer = RolloutBuffer()

        config = self.config
        n_steps: int = config["n_steps"]
        batch_size: int = config["batch_size"]
        n_epochs: int = config["n_epochs"]
        gamma: float = config["gamma"]
        gae_lambda: float = config["gae_lambda"]
        checkpoint_freq: int = config.get("checkpoint_freq", 50_000)

        global_step = getattr(self, "_resume_step", 0)
        episode_rewards: list[float] = []
        current_ep_reward = 0.0
        obs = env.reset()

        self.model.train()
        print(f"[RLTrainer] Starting training — device={device}, total_timesteps={total_timesteps:,}")

        while global_step < total_timesteps:
            # ── 1. Collect rollout ─────────────────────────────────────
            for _ in range(n_steps):
                if augmenter is not None:
                    obs = augmenter(obs)

                obs_t = self._preprocess_obs(obs, device)
                with torch.no_grad():
                    action, log_prob, value = self.model.get_action(obs_t)

                action_np = action.cpu().numpy()
                next_obs, reward, dones, _ = env.step(action_np)

                buffer.add(
                    obs=obs[0],              # (H, W, 4) — remove batch dim
                    action=int(action_np[0]),
                    reward=float(reward[0]),
                    done=bool(dones[0]),
                    value=float(value[0].cpu()),
                    log_prob=float(log_prob[0].cpu()),
                )

                current_ep_reward += float(reward[0])
                if dones[0]:
                    episode_rewards.append(current_ep_reward)
                    current_ep_reward = 0.0

                obs = next_obs
                global_step += 1

                if global_step >= total_timesteps:
                    break

            # ── 2. Compute advantages ──────────────────────────────────
            obs_t = self._preprocess_obs(obs, device)
            with torch.no_grad():
                _, last_value = self.model(obs_t)
            buffer.compute_advantages(
                last_value=float(last_value[0].cpu()),
                gamma=gamma,
                gae_lambda=gae_lambda,
            )

            # ── 3. PPO update (multiple epochs over the same rollout) ──
            all_loss_info: list[dict] = []
            for _ in range(n_epochs):
                for batch in buffer.get_batches(batch_size, device=str(device)):
                    self.train_loop(batch)
                    all_loss_info.append(self._last_loss_info)

            # ── 4. Reset buffer ────────────────────────────────────────
            buffer.clear()

            # ── 5. Logging ─────────────────────────────────────────────
            metrics = self._average_loss_info(all_loss_info)
            if episode_rewards:
                metrics["rollout/mean_ep_reward"] = float(np.mean(episode_rewards[-20:]))
                metrics["rollout/ep_count"] = len(episode_rewards)
            metrics["rollout/global_step"] = global_step

            self.logger.log(metrics, step=global_step)

            print(
                f"[{global_step:>8,}/{total_timesteps:,}] "
                f"loss={metrics['loss/total']:.4f}  "
                f"reward={metrics.get('rollout/mean_ep_reward', float('nan')):.1f}  "
                f"episodes={len(episode_rewards)}"
            )

            # ── 6. Checkpoint ──────────────────────────────────────────
            if global_step % checkpoint_freq < n_steps:
                ckpt_path = os.path.join("weights", run_name, f"step_{global_step}.pt")
                self.save_checkpoint(ckpt_path)

        # Final checkpoint + plot
        self.save_checkpoint(os.path.join("weights", run_name, "final.pt"))
        self.logger.save_plot(episode_rewards)
        self.logger.close()
        print(f"[RLTrainer] Training complete. {len(episode_rewards)} episodes finished.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_obs(obs: np.ndarray, device) -> torch.Tensor:
        """
        Convert raw env observation to a float tensor.

        obs shape from VecFrameStack: (1, H, W, 4)  uint8
        Output:                       (1, 4, H, W)  float32 in [0, 1]
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
        obs_t = obs_t.permute(0, 3, 1, 2)   # NHWC → NCHW
        return obs_t

    @staticmethod
    def _average_loss_info(infos: list[dict]) -> dict:
        """Average loss dicts from multiple mini-batch updates."""
        averaged = {}
        for key in infos[0]:
            averaged[key] = float(np.mean([d[key] for d in infos]))
        return averaged
