# test_env.py
from utils.data_reader import make_env

env = make_env(render_mode="human")   # opens a game window
obs = env.reset()
print("Obs shape:", obs.shape)        # should be (1, 240, 256, 4)

for _ in range(5000):                  # run 500 random steps
    action = env.action_space.sample()
    obs, reward, done, info = env.step([action])
    if done[0]:
        obs = env.reset()

env.close()
print("Environment works!")
