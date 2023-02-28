import random
from rsoccer_gym.Render.Render import RCGymRender

from rsoccer_gym.ssl.ssl_path_planning.navigation import Point2D, GoToPointEntry, go_to_point, abs_smallest_angle_diff, dist_to, length

import gym
import numpy as np
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree

ANGLE_TOLERANCE: float = np.deg2rad(7.5)
SPEED_TOLERANCE: float = 0.01  # m/s == 1 cm/s
DIST_TOLERANCE: float = 0.2  # m == 20 cm

class SSLPathPlanningEnv(SSLBaseEnv):
    """The SSL robot needs to reach the target point with a given angle"""

    def __init__(self, field_type=1, n_robots_yellow=0):
        super().__init__(field_type=field_type, n_robots_blue=1,
                         n_robots_yellow=n_robots_yellow, time_step=0.025)

        self.action_space = gym.spaces.Box(low=-1, high=1,  # hyp tg.
                                           shape=(4, ), dtype=np.float32)

        n_obs = 5 + 4 + 7*self.n_robots_blue + 2*self.n_robots_yellow
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs, ),
                                                dtype=np.float32)

        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10

        self.target_point: Point2D = Point2D(0, 0)
        self.target_angle: float = 0.0
        self.target_speed: float = 0.0

        self.reward_info = {
            'cumulative_dist_reward': 0,
            'cumulative_angle_reward': 0,
            'cumulative_speed_reward': 0,
            'total_reward': 0,

            'dist_error': 0,
            'angle_error': 0,
            'speed_error': 0,

            'current_speed': 0,
        }

        print('Environment initialized')
    
    def reset(self):
        self.reward_info = {
            'cumulative_dist_reward': 0,
            'cumulative_angle_reward': 0,
            'cumulative_speed_reward': 0,
            'total_reward': 0,

            'dist_error': 0,
            'angle_error': 0,
            'speed_error': 0,
            
            'current_speed': 0,
        }
        return super().reset()
    
    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_info

    def _frame_to_observations(self):
        observation = list()

        observation.append(self.norm_pos(self.target_point.x))
        observation.append(self.norm_pos(self.target_point.y))
        observation.append(np.sin(self.target_angle))
        observation.append(np.cos(self.target_angle))
        observation.append(self.norm_v(self.target_speed))

        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, action):
        field_half_length = self.field.length / 2  # x
        field_half_width = self.field.width / 2    # y

        target_x = action[0] * field_half_length
        target_y = action[1] * field_half_width
        target_angle = np.arctan2(action[2], action[3])

        entry: GoToPointEntry = GoToPointEntry()
        entry.target = Point2D(target_x * 1000.0, target_y * 1000.0)  # m to mm
        entry.target_angle = target_angle
        entry.using_prop_velocity = True

        robot = self.frame.robots_blue[0]
        angle = np.deg2rad(robot.theta)
        position = Point2D(x=robot.x * 1000.0, y=robot.y * 1000.0)

        result = go_to_point(agent_position=position,
                             agent_angle=angle,
                             entry=entry)

        return [
            Robot(
                yellow=False,
                id=0,
                v_x=result.velocity.x,
                v_y=result.velocity.y,
                v_theta=result.angular_velocity
            )
        ]

    def is_v_in_range(self, current, target) -> bool:
        return abs(current - target) <= SPEED_TOLERANCE

    def reward_function(self, robot_pos: Point2D, last_robot_pos: Point2D, robot_vel: Point2D, robot_angle: float, target_pos: Point2D, target_angle: float, target_speed: float):
        max_dist = np.sqrt(self.field.length ** 2 + self.field.width ** 2)

        last_dist_robot_to_target = dist_to(target_pos, last_robot_pos)
        dist_robot_to_target = dist_to(target_pos, robot_pos)

        last_robot_angle = np.deg2rad(self.last_frame.robots_blue[0].theta)
        last_angle_error = abs_smallest_angle_diff(last_robot_angle, target_angle)
        angle_error = abs_smallest_angle_diff(robot_angle, target_angle)

        robot_speed = length(robot_vel)

        angle_reward = 0.125 * (last_angle_error - angle_error) / np.pi
        dist_reward = 0.75 * (last_dist_robot_to_target - dist_robot_to_target) / max_dist
        speed_reward =  0.125 if self.is_v_in_range(robot_speed, target_speed) else 0.0

        self.reward_info['dist_error'] = dist_robot_to_target
        self.reward_info['angle_error'] = angle_error
        self.reward_info['speed_error'] = abs(robot_speed - target_speed)

        self.reward_info['current_speed'] = robot_speed

        if angle_error <= ANGLE_TOLERANCE:
            if dist_robot_to_target <= DIST_TOLERANCE:
                self.reward_info['total_reward'] += speed_reward
                self.reward_info['cumulative_speed_reward'] += speed_reward
                return speed_reward, True

            self.reward_info['total_reward'] += dist_reward
            self.reward_info['cumulative_dist_reward'] += dist_reward
            return dist_reward, False
        
        self.reward_info['total_reward'] += angle_reward
        self.reward_info['cumulative_angle_reward'] += angle_reward
        return angle_reward, False

    def _calculate_reward_and_done(self):
        robot = self.frame.robots_blue[0]
        last_robot = self.last_frame.robots_blue[0]

        robot_pos = Point2D(x=robot.x, y=robot.y)
        last_robot_pos = Point2D(x=last_robot.x, y=last_robot.y)
        robot_angle = np.deg2rad(robot.theta)
        target_pos = self.target_point
        target_angle = self.target_angle
        target_speed = self.target_speed

        robot_vel = Point2D(x=robot.v_x, y=robot.v_y)

        reward, done = self.reward_function(robot_pos=robot_pos,
                                            last_robot_pos=last_robot_pos,
                                            robot_vel=robot_vel,
                                            robot_angle=robot_angle,
                                            target_pos=target_pos,
                                            target_angle=target_angle,
                                            target_speed=target_speed)
        return reward, done

    def _get_initial_positions_frame(self):
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def get_random_x():
            return random.uniform(-field_half_length + 0.1,
                                  field_half_length - 0.1)

        def get_random_y():
            return random.uniform(-field_half_width + 0.1,
                                  field_half_width - 0.1)

        def get_random_theta():
            return random.uniform(0, 360)
        
        def get_random_speed():
            return random.uniform(0, self.max_v)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=get_random_x(), y=get_random_y())

        self.target_point = Point2D(x=get_random_x(), y=get_random_y())
        self.target_angle = np.deg2rad(get_random_theta())

        self.target_speed = 0.0 # get_random_speed()

        #  TODO: Move RCGymRender to another place
        self.view = RCGymRender(self.n_robots_blue,
                                self.n_robots_yellow,
                                self.field,
                                simulator='ssl',
                                angle_tolerance=ANGLE_TOLERANCE)

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
            pos_frame.robots_blue[i] = Robot(id=i, yellow=False,
                                             x=pos[0], y=pos[1], theta=get_random_theta())

        for i in range(self.n_robots_yellow):
            pos = (get_random_x(), get_random_y())
            while places.get_nearest(pos)[1] < min_gen_dist:
                pos = (get_random_x(), get_random_y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(id=i, yellow=True,
                                               x=pos[0], y=pos[1], theta=get_random_theta())

        return pos_frame
