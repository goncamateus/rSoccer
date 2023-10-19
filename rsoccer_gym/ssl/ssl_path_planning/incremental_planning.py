import numpy as np

from rsoccer_gym.ssl.ssl_path_planning.ssl_path_planning import (
    ANGLE_TOLERANCE, DIST_TOLERANCE, Point2D, SSLPathPlanningEnv,
    abs_smallest_angle_diff, dist_to)


class IncrementalPlanningEnv(SSLPathPlanningEnv):
    def _get_commands(self, action):
        robot = self.frame.robots_blue[0]
        field_half_length = self.field.length / 2  # x
        field_half_width = self.field.width / 2  # y
        action[0] = action[0] + robot.x
        action[0] = np.clip(action[0], -field_half_length, field_half_length)
        action[0] = action[0] / field_half_length
        action[1] = action[1] + robot.y
        action[1] = np.clip(action[1], -field_half_width, field_half_width)
        action[1] = action[1] / field_half_width
        return super()._get_commands(action)

    def _dist_reward(self):
        robot = self.frame.robots_blue[0]       
        robot = Point2D(x=robot.x, y=robot.y)
        actual_dist = dist_to(robot, self.target_point)
        reward = -actual_dist if actual_dist > DIST_TOLERANCE else 10
        return reward, actual_dist

    def _angle_reward(self):
        self.frame.robots_blue[0]
        action_angle = np.arctan2(self.actual_action[2], self.actual_action[3])
        target = self.target_angle
        angle_diff = abs_smallest_angle_diff(action_angle, target)
        angle_reward = -angle_diff / np.pi if angle_diff > ANGLE_TOLERANCE else 1
        return angle_reward, angle_diff
