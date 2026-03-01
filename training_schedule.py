# training_schedule.py
"""
TensorBoard:
    tensorboard --logdir logs/tensorboard
"""

import os
from torch.optim import Adam

from utils.data_reader import make_env
from utils.augment import ObsAugmenter
from models.policy import Net
from losses.ppo_loss import PPOLoss
from logs.logger import Logger
from trainer.RL_trainer import RLTrainer as RLTrainer

# ──────────────────────────────────────────────
USE_AUGMENTATION = False  # add Gaussian noise to observations
USE_WANDB = True  # log to Weights & Biases
USE_TENSORBOARD = True  # log to TensorBoard

DEVICE = "mps"  # "mps" (M1 Mac) | "cuda" (NVIDIA) | "cpu"

RESUME_CHECKPOINT = "weights/mario_ppo_50000.pt"  # set to None if starting fresh

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
config = {
    "lr": 2.5e-4,  # Adam learning rate
    "gamma": 0.9,  # discount factor
    "gae_lambda": 0.95,  # GAE smoothing (0=TD, 1=MC)
    "clip_eps": 0.2,  # PPO clip epsilon
    "n_steps": 512,  # env steps per rollout before each PPO update
    "batch_size": 64,  # mini-batch size for PPO update
    "n_epochs": 10,  # PPO update epochs per rollout
    "total_timesteps": 1_000_000,
    "checkpoint_freq": 50_000,  # save weights every N global steps
}


# ──────────────────────────────────────────────
# Wire everything up
# ──────────────────────────────────────────────
def main():
    os.makedirs("weights", exist_ok=True)

    env = make_env(render_mode=None)  # headless for training
    n_actions = env.action_space.n  # 7 with SIMPLE_MOVEMENT

    augmenter = ObsAugmenter() if USE_AUGMENTATION else None

    model = Net(n_actions=n_actions).to(DEVICE)
    loss_fn = PPOLoss(clip_eps=config["clip_eps"])
    optimizer = Adam(model.parameters(), lr=config["lr"])

    # Load run ID from the checkpoint if resuming
    resume_run_id = None
    if RESUME_CHECKPOINT:
        ckpt = torch.load(RESUME_CHECKPOINT, map_location="cpu")
        resume_run_id = ckpt.get("wandb_run_id", None)

    logger = Logger(
        use_wandb=USE_WANDB,
        use_tensorboard=USE_TENSORBOARD,
        project="mario-ppo",
        log_dir="logs/tensorboard",
        config=config,
        resume_run_id=resume_run_id,
    )

    trainer = RLTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=config,
        logger=logger,
    )

    if RESUME_CHECKPOINT:
        trainer.load_checkpoint(RESUME_CHECKPOINT)

    trainer.train(env, total_timesteps=config["total_timesteps"], augmenter=augmenter)
    env.close()


if __name__ == "__main__":
    main()
