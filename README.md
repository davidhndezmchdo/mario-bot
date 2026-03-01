# Mario Bot — Deep RL from Scratch

Teaching an agent to play **Super Mario Bros** using **Proximal Policy Optimization (PPO)** implemented from scratch in PyTorch.

---

## Project Structure

```
mario_bot/
├── pyproject.toml              # UV — dependencies and project metadata
├── training_schedule.py        # Entry point — config
├── models/
│   └── policy.py               # Net — shared CNN → actor head + critic head
├── losses/
│   └── ppo_loss.py             # Clipped surrogate + value loss + entropy bonus
├── logs/
│   └── logger.py               # Logger — WandB + TensorBoard + matplotlib
├── weights/                    # Saved checkpoints (gitignored)
├── utils/
│   ├── data_reader.py          # MarioEnv wrapper + make_env() factory
│   ├── dataset.py              # RolloutBuffer — GAE-λ advantage estimation
│   └── augment.py              # ObsAugmenter — optional Gaussian noise
└── trainer/
    ├── base_trainer.py         # Abstract Trainer — save/load checkpoint
    └── RL_trainer.py           # RLTrainer — rollout collection + PPO update loop
```

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/yourusername/mario_bot.git
cd mario_bot
uv sync
```

---

## Training

All configuration lives at the top of `training_schedule.py`. Edit the config dict, then run:

```bash
uv run training_schedule.py
```

**Key toggles:**

```python
USE_AUGMENTATION = False   # Gaussian noise on observations
USE_WANDB        = True    # Log to Weights & Biases
USE_TENSORBOARD  = True    # Log to TensorBoard

DEVICE = "mps"             # "mps" (Apple Silicon) | "cuda" (NVIDIA) | "cpu"
```

**Hyperparameters:**

| Parameter | Default | Description |
|---|---|---|
| `lr` | `2.5e-4` | Adam learning rate |
| `gamma` | `0.9` | Discount factor |
| `gae_lambda` | `0.95` | GAE smoothing (0=TD, 1=Monte Carlo) |
| `clip_eps` | `0.2` | PPO clip epsilon |
| `n_steps` | `512` | Env steps per rollout |
| `batch_size` | `64` | PPO mini-batch size |
| `n_epochs` | `10` | Update epochs per rollout |
| `total_timesteps` | `1_000_000` | Total training steps |
| `checkpoint_freq` | `50_000` | Steps between checkpoint saves |

**Monitor training:**

```bash
# TensorBoard
tensorboard --logdir logs/tensorboard

# WandB — opens automatically in browser if USE_WANDB=True
```

**Watch the trained agent play:**

```python
# In training_schedule.py, change render_mode and uncomment load_checkpoint:
env = make_env(render_mode="human")
trainer.load_checkpoint("weights/mario_ppo_final.pt")
trainer.test_loop(env, n_episodes=5)
```

---

## Architecture

### Environment Preprocessing

The raw NES emulator output goes through the following pipeline:

```
gym_super_mario_bros.make("SuperMarioBros-v0")
  → JoypadSpace(SIMPLE_MOVEMENT)     # 7 discrete actions
  → MarioEnv (custom Gymnasium wrapper)
      • RGB → grayscale  (luminance formula, no cv2)
      • Correct reset()/step() signatures for SB3 v2+
  → DummyVecEnv                      # vectorized, required by VecFrameStack
  → VecFrameStack(n=4)               # obs shape: (1, 240, 256, 4)
```

> **Why a custom wrapper?** SB3 v2+ uses `shimmy` to auto-wrap old-gym envs, which intercepts `reset()` calls and injects a `seed=` argument before our wrappers can handle it. `MarioEnv` subclasses `gymnasium.Env` directly, bypassing shimmy entirely.

### Policy Network (`Net`)

A shared CNN backbone feeds into separate actor and critic heads:

```
Input: (B, 4, 240, 256)  — 4 stacked grayscale frames

CNN backbone:
  Conv(4→32,  8×8, stride=4) + ReLU
  Conv(32→64, 4×4, stride=2) + ReLU
  Conv(64→64, 3×3, stride=1) + ReLU
  Flatten → Linear(→512) + ReLU

Actor head:  Linear(512 → 7)  → Categorical distribution → sampled action
Critic head: Linear(512 → 1)  → scalar value estimate V(s)
```

Sharing the backbone means both heads develop the same spatial feature understanding, while using roughly half the parameters of two separate networks.

### PPO Loss

```
ratio        = exp(log π_new - log π_old)
policy_loss  = -mean( min(ratio·A, clip(ratio, 1-ε, 1+ε)·A) )
value_loss   = MSE(V(s), returns)
entropy_loss = -mean(H(π))

total = policy_loss + 0.5·value_loss + 0.01·entropy_loss
```

- **Clipping** prevents destructively large policy updates — PPO's core stability trick.
- **Entropy bonus** keeps the policy from collapsing to a single action too early in training.
- **GAE-λ advantages** (`λ=0.95`) balance bias vs. variance in advantage estimates.

---

## Roadmap

### Models to Try
- [ ] Recurrent policy (LSTM backbone) — for partial observability / long-horizon memory
- [ ] Transformer-based policy — self-attention over the frame stack
- [ ] Larger CNN (Nature DQN architecture — 3 conv layers, wider channels)

### RL Algorithms to Try
- [ ] **A2C** — synchronous advantage actor-critic; simpler on-policy baseline
- [ ] **DQN** — off-policy Q-learning; compare sample efficiency vs. PPO
- [ ] **SAC** — soft actor-critic; entropy-regularised off-policy method
- [ ] **Dreamer / MBRL** — model-based RL; learn a world model to plan inside

---

## Hardware

| Machine | Device setting |
|---|---|
| Apple Silicon (M1/M2/M3) | `DEVICE = "mps"` |
| NVIDIA GPU | `DEVICE = "cuda"` |
| CPU only | `DEVICE = "cpu"` |

---

## Dependencies

Managed via `uv` / `pyproject.toml`:

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Neural network + autograd |
| `gym-super-mario-bros`, `nes-py` | NES emulator + Mario environment |
| `gymnasium`, `stable-baselines3` | `DummyVecEnv`, `VecFrameStack` wrappers |
| `shimmy` | SB3 ↔ Gymnasium compatibility layer |
| `wandb`, `tensorboard` | Training metrics logging |
| `matplotlib` | Reward curve plots |
| `numpy` | Array ops (pinned `<2.0` for SB3 compatibility) |
