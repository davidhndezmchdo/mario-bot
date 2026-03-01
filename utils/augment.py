"""
ObsAugmenter — optional observation augmentation applied during rollout collection.

Currently implements: low-intensity Gaussian noise.

WHY noise augmentation?
  Adding a small amount of pixel noise acts as a regulariser — it prevents the
  policy from memorising specific pixel patterns and encourages learning more
  robust, generalised visual features. The intensity is kept very low (~1%)
  so game objects remain clearly distinguishable.

Toggled from training_schedule.py via USE_AUGMENTATION flag.
"""

import numpy as np


class ObsAugmenter:
    def __init__(self, noise_std: float = 0.01):
        """
        Args:
            noise_std: Standard deviation of Gaussian noise added to obs.
                       Applied in [0, 1] float space. 0.01 ≈ 2-3 pixel values
                       out of 255.
        """
        self.noise_std = noise_std

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to a raw observation.

        Args:
            obs: uint8 array of any shape (e.g. (240, 256, 4))

        Returns:
            Augmented uint8 array of the same shape.
        """
        # Convert to float, add noise, clip, return as uint8
        obs_float = obs.astype(np.float32) / 255.0
        noise = np.random.normal(0.0, self.noise_std, obs_float.shape).astype(np.float32)
        obs_float = np.clip(obs_float + noise, 0.0, 1.0)
        return (obs_float * 255.0).astype(np.uint8)
