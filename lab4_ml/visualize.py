import gymnasium as gym

env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95,
               domain_randomize=False, continuous=False)

observation, info = env.reset(seed=42)

for step in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
