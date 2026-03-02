# evaluate.py
"""
Watch a trained Mario agent play.

Usage:
    uv run python evaluate.py                          # uses CHECKPOINT below
    uv run python evaluate.py weights/mario_ppo_final.pt
"""

import sys
import torch
import numpy as np

from utils.data_reader import make_env
from models.policy import Net
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# ──────────────────────────────────────────────
CHECKPOINT = "./weights/mario_ppo_final.pt"   # default — override via CLI arg
N_EPISODES = 5
DEVICE     = "mps"   # "mps" | "cuda" | "cpu"
# ──────────────────────────────────────────────


def preprocess(obs: np.ndarray, device: str) -> torch.Tensor:
    """(1, H, W, 4) uint8  →  (1, 4, H, W) float32 in [0, 1]"""
    t = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
    return t.permute(0, 3, 1, 2)


def evaluate(checkpoint_path: str, n_episodes: int, device: str):
    env = make_env(render_mode="human")
    n_actions = env.action_space.n

    model = Net(n_actions=n_actions).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    step = ckpt.get("global_step", "?")
    print(f"Loaded checkpoint: {checkpoint_path}  (trained to step {step})")
    print(f"Running {n_episodes} episode(s) — close the window to stop early.\n")

    episode_rewards = []

    with torch.no_grad():
        for ep in range(1, n_episodes + 1):
            obs = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                obs_t = preprocess(obs, device)
                logits, _ = model(obs_t)
                action = logits.argmax(dim=-1).item()
                obs, reward, dones, _ = env.step([action])
                total_reward += float(reward[0])
                done = bool(dones[0])

            episode_rewards.append(total_reward)
            print(f"  Episode {ep}: reward = {total_reward:.1f}")

    env.close()
    print(f"\nMean reward over {n_episodes} episode(s): {np.mean(episode_rewards):.2f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CHECKPOINT
    evaluate(path, N_EPISODES, DEVICE)
