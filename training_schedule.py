# training_schedule.py
"""
TensorBoard:
    tensorboard --logdir logs/<run_name>
"""

import torch
from torch.optim import Adam
from utils.data_reader import make_env
from utils.augment import ObsAugmenter
from utils.run_name import generate_run_name, save_config
from models.policy import Net
from losses.ppo_loss import PPOLoss
from utils.logger import Logger
from trainer.RL_trainer import RLTrainer

# Keys from config to embed in the auto-generated run name.
# Change this list to highlight whichever params matter for the current experiment.
NAME_KEYS = ["lr", "n_epochs", "batch_size", "gamma"]

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
config = {
    "lr": 2.5e-4,  # Adam learning rate
    "gamma": 0.9,  # discount factor
    "gae_lambda": 0.95,  # GAE smoothing (0=TD, 1=MC)
    "clip_eps": 0.2,  # PPO clip epsilon
    "value_coef": 0.25,  # value loss coefficient
    "entropy_coef": 0.01,  # entropy bonus coefficient
    "n_steps": 2048,  # env steps per rollout before each PPO update
    "batch_size": 64,  # mini-batch size for PPO update
    "n_epochs": 4,  # PPO update epochs per rollout
    "total_timesteps": 1_000_000,
    "checkpoint_freq": 50_000,  # save weights every N global steps
    #
    #
    "augment": False,
    "wandb": True,
    "tensorboard": True,
    "device": "mps",
    "resume_checkpoint": None,
}

DEVICE = config["device"]
RESUME_CHECKPOINT = config["resume_checkpoint"]

# ──────────────────────────────────────────────
# Wire everything up
# ──────────────────────────────────────────────
def main():
    env = make_env(render_mode=None)  # headless for training
    n_actions = env.action_space.n  # 7 with SIMPLE_MOVEMENT

    augmenter = ObsAugmenter() if config["augment"] is True else None

    model = Net(n_actions=n_actions).to(DEVICE)
    loss_fn = PPOLoss(
        clip_eps=config["clip_eps"],
        value_coef=config["value_coef"],
        entropy_coef=config["entropy_coef"],
    )
    optimizer = Adam(model.parameters(), lr=config["lr"])

    # ── Run name & config persistence ─────────────────────────────────
    run_name = generate_run_name(
        model_class=model.__class__.__name__,
        trainer_class=RLTrainer.__name__,
        config=config,
        name_keys=NAME_KEYS,
    )
    save_config(run_name, config)
    print(f"[Run] {run_name}")

    # ── Resume: pull wandb run ID from checkpoint ──────────────────────
    resume_run_id = None
    if RESUME_CHECKPOINT:
        ckpt = torch.load(RESUME_CHECKPOINT, map_location="cpu")
        resume_run_id = ckpt.get("wandb_run_id", None)

    logger = Logger(
        use_wandb=config["wandb"],
        use_tensorboard=config["tensorboard"],
        project="mario-ppo",
        run_name=run_name,
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

    trainer.train(
        env,
        total_timesteps=config["total_timesteps"],
        run_name=run_name,
        augmenter=augmenter,
    )
    env.close()


if __name__ == "__main__":
    main()
