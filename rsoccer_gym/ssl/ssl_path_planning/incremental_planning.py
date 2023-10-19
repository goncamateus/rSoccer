import numpy as np
from rsoccer_gym.ssl.ssl_path_planning.ssl_path_planning import SSLPathPlanningEnv


class IncrementalPlanningEnv(SSLPathPlanningEnv):
    def _get_commands(self, action):
        robot = self.frame.robots_blue[0]
        field_half_length = self.field.length / 2  # x
        field_half_width = self.field.width / 2  # y
        dist = self.time_step * self.max_v * 10
        action[:2] = action[:2] * dist
        action[0] = action[0] + robot.x
        action[0] = np.clip(action[0], -field_half_length, field_half_length)
        action[0] = action[0] / field_half_length
        action[1] = action[1] + robot.y
        action[1] = np.clip(action[1], -field_half_width, field_half_width)
        action[1] = action[1] / field_half_width
        return super()._get_commands(action)
