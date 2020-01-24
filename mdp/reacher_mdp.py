import numpy as np
import os
from pathlib import Path

root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
os.sys.path.append(root_path)

from mdp.mujoco_env.reacher import GymReacher

class ReacherMDP():
    action_dim = 2
    def __init__(self, width=48, height=48):
        """
        Args:
          width: width of the rendered image.
          height: height of the rendered image.
        """
        self.width = width
        self.height = height
        self.env = GymReacher(width=width, height=height)
        self.action_range = np.array([self.env.action_space.low, self.env.action_space.high])


    def transition_function(self, s, u):
        # clip the action
        u = np.clip(u, self.action_range[0], self.action_range[1])       
        s_next = self.env.true_dynamic(s, u)

        return s_next

    def render(self, s):
        self.env._set_state(s)
        return self.env._get_obs()

    def sample_random_state(self):
        return self.env.sample_random_state()

    def sample_random_action(self):
        return self.env.action_space.sample()
