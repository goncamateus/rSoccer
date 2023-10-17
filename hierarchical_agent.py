import gym
import rsoccer_gym
import numpy as np
from rsoccer_gym.ssl.hierarchical_go_to import SSLHierarchicalGoToEnv

env = SSLHierarchicalGoToEnv()

for i in range(10):
    env.reset()
    env.render()
    done = {"worker": False, "manager": False}
    while not done["manager"]:
        # action = np.array(
        #     [
        #         # 0,
        #         env.target_point.x / (env.field.length / 2),
        #         env.target_point.y / (env.field.width / 2),
        #         np.sin(env.target_angle),
        #         np.cos(env.target_angle),
        #         # env.target_velocity.x / 2.5,
        #         # env.target_velocity.y / 2.5,
        #     ]
        # )
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(info)
        env.render()
