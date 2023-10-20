import numpy as np

from rsoccer_gym.ssl.ssl_path_planning.ssl_path_planning import (
    ANGLE_TOLERANCE,
    DIST_TOLERANCE,
    GoToPointEntryNew,
    Point2D,
    Robot,
    SSLPathPlanningEnv,
    abs_smallest_angle_diff,
    dist_to,
    go_to_point_new,
)


class IncrementalPlanningEnv(SSLPathPlanningEnv):
    def _get_commands(self, action):
        robot = self.frame.robots_blue[0]
        field_half_length = self.field.length / 2  # x
        field_half_width = self.field.width / 2  # y
        target_x = action[0] + robot.x
        target_x = np.clip(target_x, -field_half_length, field_half_length)
        target_y = action[1] + robot.y
        target_y = np.clip(target_y, -field_half_width, field_half_width)
        target_angle = np.arctan2(action[2], action[3])
        self.actual_action = np.array(
            [
                target_x / field_half_length,
                target_y / field_half_width,
                action[2],
                action[3],
            ]
        )
        target_vel_x = 0
        target_vel_y = 0
        entry: GoToPointEntryNew = GoToPointEntryNew()
        entry.target = Point2D(target_x, target_y)
        entry.target_angle = target_angle
        entry.target_velocity = Point2D(target_vel_x, target_vel_y)
        angle = np.deg2rad(robot.theta)
        position = Point2D(x=robot.x, y=robot.y)
        vel = Point2D(x=robot.v_x, y=robot.v_y)
        self.view.set_action_target(target_x, target_y)
        self.view.set_action_angle(np.rad2deg(target_angle))
        in_distance = dist_to(entry.target, self.target_point) < DIST_TOLERANCE
        in_angle = (
            abs_smallest_angle_diff(entry.target_angle, self.target_angle)
            < ANGLE_TOLERANCE
        )
        color = 0
        if in_distance and in_angle:
            color = 3
        elif in_distance:
            color = 1
        elif in_angle:
            color = 2
        self.view.set_action_color(color)

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

    def step(self, action):
        self.steps += 1
        # Join agent action with environment actions
        commands = self._get_commands(action)
        # Send command to simulator
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        # Get Frame from simulator
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        # Calculate environment observation, reward and done condition
        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()
        self.last_action = self.actual_action
        return observation, reward, done, {}


class ContinuousPath(IncrementalPlanningEnv):
    def _continuous_reward(self):
        field_half_length = self.field.length / 2  # x
        field_half_width = self.field.width / 2  # y
        
        robot = self.frame.robots_blue[0]
        p0 = np.array([robot.x, robot.y])

        if self.last_action is None:
            p1 = p0
        else:
            last_action_x = self.last_action[0] * field_half_length
            last_action_y = self.last_action[1] * field_half_width
            p1 = np.array([last_action_x, last_action_y])

        action_x = self.actual_action[0] * field_half_length
        action_y = self.actual_action[1] * field_half_width
        p2 = np.array([action_x, action_y])

        v0 = p1 - p0
        v0 = v0 / np.linalg.norm(v0)
        
        v1 = p2 - p1
        v1 = v1 / np.linalg.norm(v1)
        
        cos = np.dot(v0, v1)
        reward = cos if cos > 0 else -1
        return reward

    def _calculate_reward_and_done(self):
        reward, done = super()._calculate_reward_and_done()
        reward += self._continuous_reward()
        return reward, done
