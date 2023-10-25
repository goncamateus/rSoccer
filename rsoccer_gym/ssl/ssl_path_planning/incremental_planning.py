import gym
import numpy as np
from collections import deque

from rsoccer_gym.ssl.ssl_path_planning.ssl_path_planning import SSLPathPlanningEnv


class IncrementalPlanningEnv(SSLPathPlanningEnv):
    def convert_action_to_target(self, action):
        robot = self.frame.robots_blue[0]
        field_half_length = self.field.length / 2  # x
        field_half_width = self.field.width / 2  # y
        target_x = action[0] + robot.x
        target_x = np.clip(target_x, -field_half_length, field_half_length)
        target_y = action[1] + robot.y
        target_y = np.clip(target_y, -field_half_width, field_half_width)
        action = np.array(
            [
                target_x / field_half_length,
                target_y / field_half_width,
                action[2],
                action[3],
            ]
        )
        return action

    def step(self, action):
        if self.steps%16==0:
            self.actual_action = self.convert_action_to_target(action)
        for _ in range(1):
            # Join agent action with environment actions
            commands = self._get_commands(self.actual_action)
            # Send command to simulator
            self.rsim.send_commands(commands)
            self.sent_commands = commands

            # Get Frame from simulator
            self.last_frame = self.frame
            self.frame = self.rsim.get_frame()

        # Calculate environment observation, reward and done condition
        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()
        if self.steps%16==0:
            self.last_action = self.actual_action
        self.steps += 1
        return observation, reward, done, {}


class ContinuousPath(IncrementalPlanningEnv):
    def __init__(self, field_type=1, n_robots_yellow=0):
        super().__init__(field_type, n_robots_yellow)
        n_obs = 6 + 7 * self.n_robots_blue + 2 * self.n_robots_yellow + 2
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )
        self.last_actions = deque([None, None], maxlen=2)


    def _frame_to_observations(self):
        observation = super()._frame_to_observations()
        if self.actual_action is None:
            normed_action = np.zeros(2)
            normed_action[0] = self.norm_pos(self.frame.robots_blue[0].x)
            normed_action[1] = self.norm_pos(self.frame.robots_blue[0].y)
        else:
            normed_action = self.norm_pos(self.actual_action[:2])
        observation = np.concatenate([observation, normed_action])
        return observation

    def _continuous_reward(self):
        field_half_length = self.field.length / 2  # x
        field_half_width = self.field.width / 2  # y

        robot = self.frame.robots_blue[0]
        if self.last_actions[0] is None:
            p0 = np.array([robot.x, robot.y])
        else:
            p0 = np.array([self.last_actions[0][0], self.last_actions[0][1]])

        if self.last_actions[1] is None:
            p1 = p0
        else:
            last_action_x = self.last_actions[1][0] * field_half_length
            last_action_y = self.last_actions[1][1] * field_half_width
            p1 = np.array([last_action_x, last_action_y])

        action_x = self.actual_action[0] * field_half_length
        action_y = self.actual_action[1] * field_half_width
        p2 = np.array([action_x, action_y])

        v0 = p1 - p0
        v0_norm = np.linalg.norm(v0)
        if v0_norm > 0:
            v0 = v0 / v0_norm

        v1 = p2 - p1
        v1_norm = np.linalg.norm(v1)
        if v1_norm > 0:
            v1 = v1 / v1_norm

        p2_on_edge = False
        if abs(abs(action_x) - field_half_length) < 0.01:
            p2_on_edge = True
        if abs(abs(action_y) - field_half_width) < 0.01:
            p2_on_edge = True
        cos = np.dot(v0, v1)
        reward = 0 if cos > 0 else 5 * cos
        reward = reward if not p2_on_edge else -10
        if np.linalg.norm(p2 - self.target_point) < 0.2:
            reward = 0
        return reward

    def _calculate_reward_and_done(self):
        reward, done = super()._calculate_reward_and_done()
        reward += self._continuous_reward()
        return reward, done

    def step(self, action):
        steps = self.steps
        next_state, reward, done, info = super().step(action)
        if steps%16==0:
            self.last_actions.append(self.last_action)        
        return next_state, reward, done, info
    
    def _get_initial_positions_frame(self):
        pos_frame = super()._get_initial_positions_frame()
        self.last_actions = deque([None, None], maxlen=2)
        return pos_frame