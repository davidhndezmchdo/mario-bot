import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from matplotlib import pyplot as plt
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# 1. Create the base environment
env = gym_super_mario_bros.make(
    "SuperMarioBros-v3", apply_api_compatibility=True, render_mode="human"
)
# 2. Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order="last")

state = env.reset()
# state, reward, terminated, truncated, info = env.step([5])

# plt.figure(figsize=(20, 16))
# for idx in range(state.shape[3]):
#     plt.subplot(1, 4, idx + 1)
#     plt.imshow(state[0][:, :, idx])
# plt.show()
