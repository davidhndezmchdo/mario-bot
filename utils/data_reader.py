# utils/data_reader.py
"""
Environment factory for Super Mario Bros.

Preprocessing pipeline:
  1. gym_super_mario_bros  — base NES emulator env
  2. JoypadSpace           — restrict to 7 SIMPLE_MOVEMENT actions
  3. MarioEnv              — self-contained Gymnasium-compatible wrapper that:
                               • resets correctly (no seed/options passthrough)
                               • converts RGB→grayscale in numpy (no cv2 dependency)
                               • returns 5-tuple from step() as SB3 v2 expects
  4. DummyVecEnv           — vectorized wrapper required by VecFrameStack
  5. VecFrameStack(n=4)    — stack last 4 frames → shape (1, 240, 256, 4)
                             channels_order="last" keeps HWC layout for the CNN

WHY a custom wrapper instead of gym.wrappers.GrayScaleObservation?
  SB3 v2+ uses shimmy to auto-wrap old gym envs, which intercepts the call
  chain and passes seed= to reset() before our wrappers can catch it.
  By implementing a standalone gymnasium.Env subclass we bypass shimmy
  entirely — SB3 sees a native Gymnasium env and needs no compatibility shim.
"""

import numpy as np
import gymnasium
from gymnasium import spaces
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


class MarioEnv(gymnasium.Env):
    """
    Standalone Gymnasium wrapper around the old-gym Mario environment.

    Handles:
      - RGB → grayscale conversion using the standard luminance formula
        (no cv2 required): Y = 0.299R + 0.587G + 0.114B
      - Correct reset() / step() signatures for SB3 v2+
      - Passes render_mode through for human viewingtandalone Gymnasium wrapper around the old-gym Mario environment.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None) -> None:
        # Build the old-gym env
        env = gym_super_mario_bros.make(
            "SuperMarioBros-v0",
            apply_api_compatibility=True,
            render_mode=render_mode,
        )

        self._env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.render_mode = render_mode

        # Observation: single grayscale channel, uint8
        h, w = 240, 256
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, 1), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(SIMPLE_MOVEMENT))

    def _to_gray(self, rgb_obs):
        """Convert (H, W, 3) uint8 RGB -> (H, W, 1) uint8 grayscale"""
        # Luminance formula
        gray = (
            0.299 * rgb_obs[:, :, 0]
            + 0.587 * rgb_obs[:, :, 1]
            + 0.114 * rgb_obs[:, :, 2]
        ).astype(np.uint8)
        return np.expand_dims(gray, axis=2)

    def reset(self, seed=None, options=None):
        # Ignore seed/options 
        obs, _ = self._env.reset()
        return self._to_gray(obs), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._to_gray(obs), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

def make_env(render_mode=None):
    """
    Build and return the preprocessed vectorized Mario environment.

    Args:
        render_mode: None for training (headless), "human" to watch gameplay.

    Returns:
        VecFrameStack env with obs shape (1, 240, 256, 4) — batch of 1.
    """
    def _make():
        return MarioEnv(render_mode=render_mode)

    env = DummyVecEnv([_make])                 # vectorize
    env = VecFrameStack(env, 4, channels_order="last")  # (1, 240, 256, 4)
    return env
