import gym
import rsoccer_gym
import numpy as np

env = gym.make("SSLHierarchical-v0")

for i in range(10):
    env.reset()
    done = {"worker": False, "manager": False}
    steps = 0
    while not done["manager"]:
        if steps == 0 or done["worker"]:
            action = env.action_space.sample()
            steps = 0
        else:
            action["worker"] = env.action_space.sample()["worker"]
        steps = (steps + 1) % env.worker_max_steps
        obs, reward, done, info = env.step(action)
        env.render()
        print(reward)
        if isinstance(done, bool):
            done = {"worker": done, "manager": done}
        if done["worker"]:
            env.reset_worker()
        env.render()
