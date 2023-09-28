import gym
import rsoccer_gym
import numpy as np
from rsoccer_gym.ssl.ssl_path_planning import SSLPathPlanningEnv

env = SSLPathPlanningEnv()
env.reset()
env.render()
done = False
while not done:
    action = np.array([0.0, 0.0, 0.0, 0.0])
    obs, reward, done, info = env.step(action)
    env.render()