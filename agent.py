import gym
import rsoccer_gym
import numpy as np
from rsoccer_gym.ssl.ssl_path_planning import SSLPathPlanningEnv

env = SSLPathPlanningEnv()
for i in range(10):
    env.reset()
    env.render()
    done = False
    while not done:
        action = np.array(
            [
                env.target_point.x / (env.field.length / 2),
                env.target_point.y / (env.field.width / 2),
                np.sin(env.target_angle),
                np.cos(env.target_angle),
                env.target_velocity.x / 2.5,
                env.target_velocity.y / 2.5,
            ]
        )
        obs, reward, done, info = env.step(action)
        print(reward)
        env.render()
