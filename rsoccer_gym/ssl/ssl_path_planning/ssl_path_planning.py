import random
from typing import List

import gym
import numpy as np

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Render.Render import RCGymRender
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.ssl.ssl_path_planning.navigation import *
from rsoccer_gym.Utils import KDTree

ANGLE_TOLERANCE: float = np.deg2rad(5)  # 5 degrees
SPEED_MIN_TOLERANCE: float = 0.05  # m/s == 5 cm/s
SPEED_MAX_TOLERANCE: float = 0.3  # m/s == 30 cm/s
DIST_TOLERANCE: float = 0.05  # m == 5 cm

class SSLPathPlanningEnv(SSLBaseEnv):
    """The SSL robot needs to reach the target point with a given angle"""

    def __init__(
        self,
        field_type=1,
        n_robots_yellow=0,
    ):
        super().__init__(
            field_type=field_type,
            n_robots_blue=1,
            n_robots_yellow=n_robots_yellow,
            time_step=0.025,
        )

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32  # hyp tg.
        )

        n_obs = 6 + 7 * self.n_robots_blue + 2 * self.n_robots_yellow
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )

        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10

        self.target_point: Point2D = Point2D(0, 0)
        self.target_angle: float = 0.0
        self.target_velocity: Point2D = Point2D(0, 0)

        self.actual_action = None

        self.reward_info = {
            "cumulative_dist_reward": 0,
            "cumulative_angle_reward": 0,
            "cumulative_velocity_reward": 0,
            "total_reward": 0,
            "dist_error": 0,
            "angle_error": 0,
            "velocity_error": 0,
            "current_speed": 0,
            "current_velocity_x": 0,
            "current_velocity_y": 0,
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

    def _get_commands(self, action):
        robot = self.frame.robots_blue[0]
        field_half_length = self.field.length / 2  # x
        field_half_width = self.field.width / 2  # y

        target_x = action[0] * field_half_length
        target_y = action[1] * field_half_width
        target_angle = np.arctan2(action[2], action[3])
        target_vel_x = action[4] * MAX_VELOCITY
        target_vel_y = action[5] * MAX_VELOCITY
        entry: GoToPointEntryNew = GoToPointEntryNew()
        entry.target = Point2D(target_x, target_y)
        entry.target_angle = target_angle
        entry.target_velocity = Point2D(target_vel_x, target_vel_y)
        self.actual_action = entry
        angle = np.deg2rad(robot.theta)
        position = Point2D(x=robot.x, y=robot.y)
        vel = Point2D(x=robot.v_x, y=robot.v_y)
        self.view.set_action_target(target_x, target_y)
        self.view.set_action_angle(np.rad2deg(target_angle))

        result = go_to_point_new(
            agent_position=position, agent_vel=vel, agent_angle=angle, entry=entry
        )
        return [
            Robot(
                yellow=False,
                id=0,
                v_x=result.velocity.x,
                v_y=result.velocity.y,
                v_theta=result.angular_velocity,
            )
        ]

    def is_v_in_range(self, current, target) -> bool:
        return abs(current - target) <= self.SPEED_TOLERANCE

    def step(self, action):
        self.steps += 1
        # Join agent action with environment actions
        commands: List[Robot] = self._get_commands(action)
        # Send command to simulator
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        # Get Frame from simulator
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        # Calculate environment observation, reward and done condition
        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()

        return observation, reward, done, {}

    def _dist_reward(self):
        action = self.actual_action.target
        target = self.target_point
        actual_dist = dist_to(action, target)
        return -actual_dist, actual_dist

    def _angle_reward(self):
        action = self.actual_action.target_angle
        target = self.target_angle
        return abs_smallest_angle_diff(action, target)

    def _calculate_reward_and_done(self):
        done = False
        reward = 0
        max_angle = np.pi
        dist_reward, distance = self._dist_reward()
        angle_reward = self._angle_reward()
        robot_dist = np.linalg.norm(
            np.array(
                [
                    self.frame.robots_blue[0].x - self.target_point.x,
                    self.frame.robots_blue[0].y - self.target_point.y,
                ]
            )
        )
        robot_vel_error = np.linalg.norm(
            np.array(
                [
                    self.frame.robots_blue[0].v_x - self.target_velocity.x,
                    self.frame.robots_blue[0].v_y - self.target_velocity.y,
                ]
            )
        )
        # print(f"dist_reward: {dist_reward < DIST_TOLERANCE} | robot_dist: {robot_dist < DIST_TOLERANCE} | angle: {angle_reward < ANGLE_TOLERANCE} | vel: {robot_vel_error < self.SPEED_TOLERANCE}")
        if (
            distance < DIST_TOLERANCE
            and angle_reward < ANGLE_TOLERANCE
            and robot_vel_error < self.SPEED_TOLERANCE
            and robot_dist < DIST_TOLERANCE
        ):
            done = True
            reward = 100
        else:
            reward = dist_reward - (angle_reward / max_angle)
        return reward, done

    def _get_initial_positions_frame(self):
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def get_random_x():
            return random.uniform(-field_half_length + 0.1, field_half_length - 0.1)

        def get_random_y():
            return random.uniform(-field_half_width + 0.1, field_half_width - 0.1)

        def get_random_theta():
            return random.uniform(0, 360)

        def get_random_speed():
            return random.uniform(0, self.max_v)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=get_random_x(), y=get_random_y())

        self.target_point = Point2D(x=get_random_x(), y=get_random_y())
        self.target_angle = np.deg2rad(get_random_theta())
        random_speed: float = get_random_speed()
        random_velocity_direction: float = np.deg2rad(get_random_theta())

        self.target_velocity = Point2D(
            x=random_speed * np.cos(random_velocity_direction),
            y=random_speed * np.sin(random_velocity_direction),
        )

        # Adjust speed tolerance according to target velocity
        target_speed_norm = np.sqrt(
            self.target_velocity.x**2 + self.target_velocity.y**2
        )
        self.SPEED_TOLERANCE = (
            SPEED_MIN_TOLERANCE
            + (SPEED_MAX_TOLERANCE - SPEED_MIN_TOLERANCE)
            * target_speed_norm
            / self.max_v
        )

        #  TODO: Move RCGymRender to another place
        self.view = RCGymRender(
            self.n_robots_blue,
            self.n_robots_yellow,
            self.field,
            simulator="ssl",
            angle_tolerance=ANGLE_TOLERANCE,
        )

        self.view.set_action_target(self.target_point.x, self.target_point.y)
        self.view.set_action_angle(np.rad2deg(self.target_angle))
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

        self.view.set_target(self.target_point.x, self.target_point.y)
        self.view.set_target_angle(np.rad2deg(self.target_angle))

        return pos_frame


if __name__ == "__main__":
    env = SSLPathPlanningEnv()
    env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(reward)
        env.render()
