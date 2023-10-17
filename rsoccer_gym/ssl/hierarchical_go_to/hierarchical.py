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
                    low=-1, high=1, shape=(2,), dtype=np.float32  # hyp tg.
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
        self.worker_max_steps = 5
        self.worker_steps = 0

        # Initializations
        self.target_point = np.zeros(2)
        self.target_angle = 0.0
        # self.target_velocity = np.zeros(2)

        self.reward_info = {
            "reward_worker/dist": 0,
            "reward_worker/angle": 0,
            "reward_worker/speed": 0,
            "reward_manager/dist": 0,
            "reward_manager/continuous": 0,
            # "reward_manager/speed": 0,
            "reward_objective": 0,
            "reward_worker/sub_objective": 0,
            "reward_worker/reward_total": 0,
            "reward_manager/reward_total": 0,
        }

        print("Environment initialized")

    def _get_commands(self, actions):
        commands = []
        v_x, v_y, v_theta = self.convert_actions(actions)
        cmd = Robot(yellow=False, id=0, v_x=v_x, v_y=v_y, v_theta=v_theta)
        commands.append(cmd)
        return commands

    def convert_actions(self, action):
        """Denormalize, clip to absolute max and convert to local"""
        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        angle = np.deg2rad(self.frame.robots_blue[0].theta)
        # Convert to local
        v_x, v_y = v_x * np.cos(angle) + v_y * np.sin(angle), -v_x * np.sin(
            angle
        ) + v_y * np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x, v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x * c, v_y * c

        return v_x, v_y, v_theta

    def _frame_to_observations(self):
        observation = list()

        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_pos(self.target_point[0]))
        observation.append(self.norm_pos(self.target_point[1]))
        observation.append(np.sin(self.target_angle))
        observation.append(np.cos(self.target_angle))
        # observation.append(self.norm_v(self.target_velocity.x))
        # observation.append(self.norm_v(self.target_velocity.y))

        observation.append(self.norm_pos(self.frame.robots_blue[0].x))
        observation.append(self.norm_pos(self.frame.robots_blue[0].y))
        observation.append(np.sin(np.deg2rad(self.frame.robots_blue[0].theta)))
        observation.append(np.cos(np.deg2rad(self.frame.robots_blue[0].theta)))
        observation.append(self.norm_v(self.frame.robots_blue[0].v_x))
        observation.append(self.norm_v(self.frame.robots_blue[0].v_y))
        observation.append(self.norm_w(self.frame.robots_blue[0].v_theta))

        return np.array(observation, dtype=np.float32)

    def _worker_observation(self):
        obs = self._frame_to_observations()
        obs = np.concatenate([obs, self.manager_target_pos])
        return obs

    def _worker_reward(self):
        robot_pos = np.array([self.frame.robots_blue[0].x, self.frame.robots_blue[0].y])
        robot_to_target = self.manager_target_pos - robot_pos
        ball_pos = np.array([self.frame.ball.x, self.frame.ball.y])
        robot_to_ball = ball_pos - robot_pos
        robot_to_ball_angle = np.arctan2(robot_to_ball[1], robot_to_ball[0])
        robot_to_target_dist = np.linalg.norm(robot_to_target)
        robot_to_ball_angle_diff = np.abs(
            self.frame.robots_blue[0].theta - robot_to_ball_angle
        )
        robot_to_ball_angle_diff = np.minimum(
            robot_to_ball_angle_diff,
            2 * np.pi - robot_to_ball_angle_diff,
        )

        robot_to_target = robot_to_target / robot_to_target_dist
        robot_vel = np.array(
            [self.frame.robots_blue[0].v_x, self.frame.robots_blue[0].v_y]
        )
        robot_vel = robot_vel / np.linalg.norm(robot_vel)
        vel_cos_diff = np.dot(robot_to_target, robot_vel)

        dist_reward = -robot_to_target_dist
        angle_reward = robot_to_ball_angle_diff / np.pi
        speed_reward = vel_cos_diff

        return dist_reward, angle_reward, speed_reward

    def _convert_manager_action(self, action):
        self.manager_last_target_pos = self.manager_target_pos
        robot_pos = np.array([self.frame.robots_blue[0].x, self.frame.robots_blue[0].y])
        dist = action[0] * self.worker_max_steps * self.time_step * self.max_v
        angle = action[1] * np.pi
        self.manager_target_pos = robot_pos + np.array(
            [dist * np.cos(angle), dist * np.sin(angle)]
        )
        self.manager_target_pos = np.clip(
            self.manager_target_pos,
            [-self.field.length / 2, -self.field.width / 2],
            [self.field.length / 2, self.field.width / 2],
        )
        self.view.set_action_target(
            self.manager_target_pos[0], self.manager_target_pos[1]
        )
        ball_pos = np.array([self.frame.ball.x, self.frame.ball.y])
        robot_to_ball = ball_pos - robot_pos
        robot_to_ball_angle = np.arctan2(robot_to_ball[1], robot_to_ball[0])
        self.view.set_action_angle(np.rad2deg(robot_to_ball_angle))
        smallest_angle_diff = np.abs(
            np.deg2rad(self.frame.robots_blue[0].theta) - robot_to_ball_angle
        )
        abs_smallest_angle_diff = np.minimum(
            smallest_angle_diff,
            2 * np.pi - smallest_angle_diff,
        )
        color = 0
        in_angle = abs_smallest_angle_diff < self.angle_tolerance
        target_diff = np.linalg.norm(self.manager_target_pos - self.target_point)
        in_dist = target_diff < self.dist_tolerance
        if in_angle and in_dist:
            color = 1
        elif in_angle:
            color = 2
        elif in_dist:
            color = 3

        self.view.set_action_color(color)

    def _manager_reward(self):
        robot_pos = np.array([self.frame.robots_blue[0].x, self.frame.robots_blue[0].y])
        robot_to_last_manager = self.manager_last_target_pos - robot_pos
        robot_to_last_manager = robot_to_last_manager / np.linalg.norm(
            robot_to_last_manager
        )

        manager_to_target = self.target_point - self.manager_target_pos
        manager_to_target_dist = np.linalg.norm(manager_to_target)
        manager_to_target = manager_to_target / manager_to_target_dist

        cosine = np.dot(robot_to_last_manager, manager_to_target)

        # robot_vel = np.array(
        #     [self.frame.robots_blue[0].v_x, self.frame.robots_blue[0].v_y]
        # )
        # robot_vel = robot_vel / np.linalg.norm(robot_vel)
        # vel_cos_diff = np.dot(robot_to_target, robot_vel)

        dist_reward = -manager_to_target_dist
        continuous_reward = cosine if cosine > 0 else -1
        # vel_reward = vel_cos_diff

        return dist_reward, continuous_reward

    def step(self, action):
        self.steps += 1
        reward = {"worker": 0, "manager": 0}

        worker_action = action["worker"]
        # Join agent action with environment actions
        commands = self._get_commands(worker_action)
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
        self.reward_info["reward_worker/dist"] += worker_dist_reward
        self.reward_info["reward_worker/angle"] += worker_angle_reward
        self.reward_info["reward_worker/speed"] += worker_speed_reward
        self.reward_info["reward_worker/reward_total"] += worker_reward
        reward["worker"] = worker_reward

        worker_done = False
        worker_dist = -worker_dist_reward
        if worker_dist < self.dist_tolerance:
            print(
                colorize("Worker reached manager!", "blue", bold=False, highlight=False)
            )
            worker_done = True
            self.reward_info["reward_worker/sub_objective"] = 1
            self.reward_info["reward_worker/reward_total"] += 1000
            reward["worker"] = 1000

        self._convert_manager_action(action["manager"])
        # Calculate manager reward and manager done condition
        manager_reward = self._manager_reward()
        # manager_dist_reward, manager_vel_reward = manager_reward
        manager_dist_reward, manager_continuous_reward = manager_reward
        manager_reward = np.sum(manager_reward)
        self.reward_info["reward_manager/dist"] = manager_dist_reward
        self.reward_info["reward_manager/continuous"] = manager_continuous_reward
        # self.reward_info["reward_manager/speed"] = manager_vel_reward
        self.reward_info["reward_manager/reward_total"] = manager_reward
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
                self.reward_info["reward_worker/reward_total"] += 1000
                reward["worker"] = 1000
                self.reward_info["reward_manager/reward_total"] += 1000
                reward["manager"] = 1000

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
        state = {
            "worker": self._worker_observation(),
            "manager": self._frame_to_observations(),
        }
        return state, reward, done, info

    def reset_worker(self):
        self.worker_steps = 0
        self.reward_info.update(
            {
                "reward_worker/dist": 0,
                "reward_worker/angle": 0,
                "reward_worker/speed": 0,
                "reward_worker/sub_objective": 0,
                "reward_worker/reward_total": 0,
            }
        )
        state = self._frame_to_observations()
        return state

    def reset(self):
        obs = super().reset()
        self.manager_target_pos = np.array(
            [self.frame.robots_blue[0].x + 0.5, self.frame.robots_blue[0].y + 0.5]
        )
        self.reset_worker()
        self.reward_info.update(
            {
                "reward_manager/dist": 0,
                "reward_manager/continuous": 0,
                # "reward_manager/speed": 0,
                "reward_manager/reward_total": 0,
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

        self.view = RCGymRender(
            self.n_robots_blue,
            self.n_robots_yellow,
            self.field,
            simulator="ssl",
            angle_tolerance=self.angle_tolerance,
        )

        self.view.set_target(self.target_point[0], self.target_point[1])
        self.view.set_target_angle(np.rad2deg(self.target_angle))

        min_gen_dist = 0.2

        places = KDTree()
        places.insert((self.target_point[0], self.target_point[1]))
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
