import math
import random
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv
from rsoccer_gym.Utils import KDTree


class VSSTargetEnv(VSSBaseEnv):
    """This environment controls a single robot in a VSS soccer League 3v3 match 


        Description:
        Observation:
            Type: Box(4)
            Normalized Bounds to [-1.25, 1.25]
            Num             Observation normalized  
            0               id i Blue Robot X
            1               id i Blue Robot Y
            2               Objective X
            3               Objective Y
        Actions:
            Type: Box(2, )
            Num     Action
            0       id 0 Blue Left Wheel Speed  (%)
            1       id 0 Blue Right Wheel Speed (%)
        Reward:
            Sum of Rewards:
                Reach objective
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            Reach objective
    """

    def __init__(self):
        super().__init__(field_type=0, n_robots_blue=1, n_robots_yellow=0,
                         time_step=0.025)

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(2, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(7, ), dtype=np.float32)

        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.actions: Dict = None
        self.reward_shaping_total = None
        self.v_wheel_deadzone = 0.05
        self.ball_dist_scale = np.linalg.norm([0.2, 0.2])

        self.ou_actions = []
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(self.action_space, dt=self.time_step)
            )

        print('Environment initialized')

    def reset(self):
        self.actions = None
        self.reward_shaping_total = None
        self.previous_ball_potential = None
        for ou in self.ou_actions:
            ou.reset()
        return super().reset()

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total

    def _frame_to_observations(self):
        observation = []
        observation.append(self.norm_pos(self.frame.robots_blue[0].x))
        observation.append(self.norm_pos(self.frame.robots_blue[0].y))
        observation.append(self.norm_v(self.frame.robots_blue[0].v_x))
        observation.append(self.norm_v(self.frame.robots_blue[0].v_y))
        observation.append(self.norm_w(self.frame.robots_blue[0].v_theta))
        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        self.actions[0] = actions
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
        commands.append(Robot(yellow=False, id=0, v_wheel0=v_wheel0,
                              v_wheel1=v_wheel1))

        # Send random commands to the other robots
        for i in range(1, self.n_robots_blue):
            actions = self.ou_actions[i].sample()
            self.actions[i] = actions
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
            commands.append(Robot(yellow=False, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))
        for i in range(self.n_robots_yellow):
            actions = self.ou_actions[self.n_robots_blue+i].sample()
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
            commands.append(Robot(yellow=True, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))

        return commands

    def _calculate_reward_and_done(self):
        reward = 0
        goal = False
        w_energy = 2e-4
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {'goal': 0, 'dist_ball': 0, 'energy': 0}

        if self.last_frame is not None:
            # Calculate Energy penalty
            energy_penalty = self.__energy_penalty()
            robot = np.array([self.frame.robots_blue[0].x,
                              self.frame.robots_blue[0].y])
            ball = np.array([self.frame.ball.x, self.frame.ball.y])
            dist_to_ball = np.linalg.norm(robot - ball)
            reward_dist = -dist_to_ball
            if self.frame.ball.v_x != 0:
                goal = True
                reward += 10
                self.reward_shaping_total['goal'] += 1
            reward += energy_penalty*w_energy + reward_dist
            self.reward_shaping_total['final_dist'] = dist_to_ball
            self.reward_shaping_total['dist_ball'] += reward_dist
            self.reward_shaping_total['energy'] += energy_penalty


        return reward, goal

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        def theta(): return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=x(), y=y())

        min_dist = 0.1

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        
        for i in range(self.n_robots_blue):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        for i in range(self.n_robots_yellow):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame

    def _actions_to_v_wheels(self, actions):
        left_wheel_speed = actions[0] * self.max_v
        right_wheel_speed = actions[1] * self.max_v

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -self.max_v, self.max_v
        )

        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0

        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0

        # Convert to rad/s
        left_wheel_speed /= self.field.rbt_wheel_radius
        right_wheel_speed /= self.field.rbt_wheel_radius

        return left_wheel_speed, right_wheel_speed

    def __energy_penalty(self):
        '''Calculates the energy penalty'''

        en_penalty_1 = abs(self.sent_commands[0].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel1)
        energy_penalty = - (en_penalty_1 + en_penalty_2)
        return energy_penalty

    def __ball_dist_rw(self):
        assert(self.last_frame is not None)

        # Calculate previous ball dist
        last_ball = self.last_frame.ball
        last_robot = self.last_frame.robots_blue[0]
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        last_robot_pos = np.array([last_robot.x, last_robot.y])
        last_ball_dist = np.linalg.norm(last_robot_pos - last_ball_pos)

        # Calculate new ball dist
        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        ball_pos = np.array([ball.x, ball.y])
        robot_pos = np.array([robot.x, robot.y])
        ball_dist = np.linalg.norm(robot_pos - ball_pos)

        ball_dist_rw = last_ball_dist - ball_dist

        return ball_dist_rw
