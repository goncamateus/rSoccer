import random
from copy import deepcopy

import gym
import numpy as np

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Render.Render import RCGymRender
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree, colorize


class SSLHierarchicalGoToEnv(SSLBaseEnv):
    """The SSL robot needs to reach the target point with a given angle"""

    def __init__(
        self,
        field_type=1,
        n_obstacles=0,
        angle_tolerance=np.deg2rad(2.5),
        speed_tolerance=0.01,
        dist_tolerance=0.025,
    ):
        super().__init__(
            field_type=field_type,
            n_robots_blue=1,
            n_robots_yellow=n_obstacles,
            time_step=0.025,
        )
        self.action_space = gym.spaces.Dict(
            {
                "worker": gym.spaces.Box(
                    low=-1, high=1, shape=(4,), dtype=np.float32  # hyp tg.
                ),
                "manager": gym.spaces.Box(
                    low=-1, high=1, shape=(4,), dtype=np.float32  # hyp tg.
                ),
            }
        )

        n_obs = 6 + 7 * self.n_robots_blue + 2 * self.n_robots_yellow
        self.observation_space = gym.spaces.Dict(
            {
                "worker": gym.spaces.Box(
                    low=-self.NORM_BOUNDS,
                    high=self.NORM_BOUNDS,
                    shape=(n_obs,),
                    dtype=np.float32,
                ),
                "manager": gym.spaces.Box(
                    low=-self.NORM_BOUNDS,
                    high=self.NORM_BOUNDS,
                    shape=(n_obs,),
                    dtype=np.float32,
                ),
            }
        )

        # Bounds
        self.max_v = 2.5
        self.max_w = 10
        self.angle_tolerance = angle_tolerance
        self.speed_tolerance = speed_tolerance
        self.dist_tolerance = dist_tolerance
        # Steps needed to reach 1m at max speed -> 0.4s in real time
        self.worker_max_steps = (1 / self.max_v) / self.time_step
        self.worker_steps = 0

        # Initializations
        self.target_point = np.zeros(2)
        self.target_angle = 0.0
        self.target_velocity = np.zeros(2)

        self.reward_info = {
            "reward_dist_worker": 0,
            "reward_angle_worker": 0,
            "reward_speed_worker": 0,
            "reward_dist_manager": 0,
            "reward_speed_manager": 0,
            "reward_objective": 0,
            "reward_subobjective": 0,
            "reward_worker": 0,
            "reward_manager": 0,
        }

        print("Environment initialized")

    def _frame_to_observations(self):
        observation = list()

        observation.append(self.norm_pos(self.target_point.x))
        observation.append(self.norm_pos(self.target_point.y))
        observation.append(np.sin(self.target_angle))
        observation.append(np.cos(self.target_angle))
        observation.append(self.norm_v(self.target_velocity.x))
        observation.append(self.norm_v(self.target_velocity.y))

        observation.append(self.norm_pos(self.frame.robots_blue[0].x))
        observation.append(self.norm_pos(self.frame.robots_blue[0].y))
        observation.append(np.sin(np.deg2rad(self.frame.robots_blue[0].theta)))
        observation.append(np.cos(np.deg2rad(self.frame.robots_blue[0].theta)))
        observation.append(self.norm_v(self.frame.robots_blue[0].v_x))
        observation.append(self.norm_v(self.frame.robots_blue[0].v_y))
        observation.append(self.norm_w(self.frame.robots_blue[0].v_theta))

        return np.array(observation, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = {"worker": 0, "manager": 0}

        worker_action = action["worker"]
        self._get_commands(worker_action)
        # Join agent action with environment actions
        commands = self._get_commands(action)
        # Send command to simulator
        self.rsim.send_commands(commands)
        self.sent_commands = commands
        # Get Frame from simulator
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()
        robot_pos = np.array([self.frame.robots_blue[0].x, self.frame.robots_blue[0].y])

        # Calculate worker reward and worker done condition
        worker_reward = self._worker_reward()
        worker_dist_reward, worker_angle_reward, worker_speed_reward = worker_reward
        worker_reward = np.sum(worker_reward)
        self.reward_info["reward_dist_worker"] += worker_dist_reward
        self.reward_info["reward_angle_worker"] += worker_angle_reward
        self.reward_info["reward_speed_worker"] += worker_speed_reward
        self.reward_info["reward_worker"] += worker_reward
        reward["worker"] = worker_reward

        worker_done = False
        worker_dist = np.linalg.norm(robot_pos - self.manager_target_pos)
        if worker_dist < self.dist_tolerance:
            print(
                colorize("Worker reached manager!", "blue", bold=False, highlight=False)
            )
            worker_done = True
            self.reward_info["reward_subobjective"] = 1

        manager_action = self.convert_manager_action(action["manager"])
        # Calculate manager reward and manager done condition
        manager_dist_reward, manager_vel_reward = self._manager_reward(manager_action)
        manager_reward = manager_dist_reward + manager_vel_reward
        self.reward_info["reward_dist_manager"] = manager_dist_reward
        self.reward_info["reward_speed_manager"] = manager_vel_reward
        self.reward_info["reward_manager"] = manager_reward
        reward["manager"] = manager_reward

        manager_done = False
        manager_dist = np.linalg.norm(self.manager_target_pos - self.target_point)
        if manager_dist < self.dist_tolerance:
            print(colorize("Manager on target!", "blue", bold=False, highlight=False))
            robot_to_target = np.linalg.norm(robot_pos - self.target_point)
            if robot_to_target < self.dist_tolerance:
                manager_done = True
                print(colorize("REACHED TARGET!", "green", bold=True, highlight=True))
                self.reward_info["reward_objective"] = 1

        if self.worker_steps >= self.worker_max_steps and not manager_done:
            print(
                colorize(
                    "Worker did not reached manager!", "red", bold=True, highlight=False
                )
            )
            worker_done = True
        info = deepcopy(self.reward_info)
        if manager_done:
            worker_done = True

        done = {"worker": worker_done, "manager": manager_done}
        worker_state = self._frame_to_observations()
        manager_state = self._frame_to_observations()
        state = {"worker": worker_state, "manager": manager_state}
        return state, reward, done, info

    def reset_worker(self):
        self.worker_steps = 0
        self.reward_info.update(
            {
                "reward_dist_worker": 0,
                "reward_angle_worker": 0,
                "reward_speed_worker": 0,
                "reward_subobjective": 0,
                "reward_worker": 0,
            }
        )
        state = self._frame_to_observations()
        return state

    def reset(self):
        obs = super().reset()
        self.manager_target_pos = np.array(
            [self.frame.robots_blue[0].x, self.frame.robots_blue[0].y]
        )
        self.reset_worker()
        self.reward_info.update(
            {
                "reward_dist_manager": 0,
                "reward_speed_manager": 0,
                "reward_manager": 0,
                "reward_objective": 0,
            }
        )
        state = {"worker": obs, "manager": obs}
        return state

    def _get_initial_positions_frame(self):
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def get_random_x():
            return random.uniform(-field_half_length + 0.1, field_half_length - 0.1)

        def get_random_y():
            return random.uniform(-field_half_width + 0.1, field_half_width - 0.1)

        def get_random_theta():
            return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=get_random_x(), y=get_random_y())

        self.target_point = np.array([get_random_x(), get_random_y()])
        self.target_velocity = np.zeros(2)

        #  TODO: Move RCGymRender to another place
        self.view = RCGymRender(
            self.n_robots_blue,
            self.n_robots_yellow,
            self.field,
            simulator="ssl",
            angle_tolerance=self.angle_tolerance,
        )

        self.view.set_target(self.target_point.x, self.target_point.y)
        self.view.set_target_angle(np.rad2deg(self.target_angle))

        min_gen_dist = 0.2

        places = KDTree()
        places.insert((self.target_point.x, self.target_point.y))
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        for i in range(self.n_robots_blue):
            pos = (get_random_x(), get_random_y())

            while places.get_nearest(pos)[1] < min_gen_dist:
                pos = (get_random_x(), get_random_y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(
                id=i, yellow=False, x=pos[0], y=pos[1], theta=get_random_theta()
            )

        for i in range(self.n_robots_yellow):
            pos = (get_random_x(), get_random_y())
            while places.get_nearest(pos)[1] < min_gen_dist:
                pos = (get_random_x(), get_random_y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(
                id=i, yellow=True, x=pos[0], y=pos[1], theta=get_random_theta()
            )
        return pos_frame
