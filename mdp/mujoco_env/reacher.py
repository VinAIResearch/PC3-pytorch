import cv2
import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
import scipy.misc


class GymReacher(mujoco_env.MujocoEnv, utils.EzPickle):

    angle_range = [[-np.pi, -np.pi], [np.pi, np.pi]]
    velocity_range = [[-50, -30], [50, 30]]

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        utils.EzPickle.__init__(self)
        self.easy_cost = True
        self.width = kwargs['width']
        self.height = kwargs['height']
        self.random_start = True
        self.random_target = True
        self.prev_obs = None #np.zeros([64, 64, 3])
        print(os.getcwd())
        assets_dir = os.path.join(os.getcwd(), "mdp", "mujoco_env", "assets", "reacher.xml")
        mujoco_env.MujocoEnv.__init__(self, assets_dir, 2)
        

    def reward(self, x, a):
        if self.easy_cost:
            reward_dist = - np.square(x).sum()
            reward_ctrl = - np.square(a).sum()
            dist = np.linalg.norm(x)
        else:
            reward_dist = - np.linalg.norm(x)
            reward_ctrl = - np.square(a).sum()
            dist = -reward_dist
        return reward_dist + reward_ctrl, {'distance' : dist}

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward, info = self.reward(vec, a)
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, info

    def get_start(self):
        if self.random_start:
            start = np.random.uniform(low=-np.pi, high=np.pi, size=self.model.nq)
        else:
            start = np.zeros(self.model.nq)
        return start

    def create_goal(self):
        if self.random_target:
            goal = np.array([self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-0.2, 0.2)])
        else:
            goal = self.default_goal
        return goal

    def set_goal(self, goal):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        qpos[-2:] = goal

    def _set_state(self, state):
        # Concatenate state dim=4: [qpos[:2], qvel[:2]]    
        new_qpos = state[:2]
        new_qvel = state[2:]
        qpos_copy = np.copy(self.sim.data.qpos)
        qvel_copy = np.copy(self.sim.data.qvel)
        qpos_copy[:2] = new_qpos
        qvel_copy[:2] = new_qvel
        # No inplace change allowed
        # self.sim.data.qpos[:2] = new_qpos
        # self.sim.data.qvel[:2] = new_qvel
        self.set_state(qpos_copy, qvel_copy)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -90.0
        self.viewer.cam.distance = 0.6

    def reset_model(self):
        if self.random_start:
            qpos = self.np_random.uniform(low=-np.pi, high=np.pi, size=self.model.nq) + self.init_qpos
        else:
            qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        self.goal = self.create_goal()
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        self.prev_obs = None #np.zeros([64, 64, 3])
        obs = self._get_obs()
        return obs
   

    def _get_obs(self, mode='rgb'):
        img = self.render(mode='rgb_array')
        resized = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_LINEAR)
        if mode == 'rgb':
            return resized/ 255
        else:
            return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    def get_state(self):
        theta = self.sim.data.qpos.flat[:2]
        # obs = np.concatenate([
        #     np.cos(theta),
        #     np.sin(theta),
        #     self.sim.data.qpos.flat[2:],
        #     self.sim.data.qvel.flat[:2],
        #     self.get_body_com("fingertip") - self.get_body_com("target")
        # ])[:-1] + np.concatenate([np.zeros(4), np.random.normal(size=2, scale=0.01), np.zeros(4)])
        # return obs

        return np.concatenate([
            self.sim.data.qpos.flat[:2],
            self.sim.data.qvel.flat[:2],
        ])
    
    def get_goal(self):
        return self.sim.data.qpos.flat[2:]

    def sample_random_state(self):
        qpos = np.random.uniform(self.angle_range[0], self.angle_range[1])
        qvel = np.random.uniform(self.velocity_range[0], self.velocity_range[1])
        state = np.concatenate([qpos, qvel])
        return state

    def true_dynamic(self, state, action):
        # Return the next state?
        old_state = self.get_state()
        self._set_state(state)
        next_state, _, _, _ = self.step(action)
        # U should skip this?
        # self._set_state(old_state)
        return self.get_state()




if __name__ == '__main__':
    env = GymReacher(width=48, height=48)
    state = env.reset()
    
    print("Reset state: ", env.get_state())
    sampled_state = env.sample_random_state()
    print("Sampled state:", sampled_state)
    action = env.action_space.sample()
    print("Sampled action:", action)
    next_state = env.true_dynamic(sampled_state, action)
    print("Next state:", next_state)
    print("Current state:", sampled_state)

    # Test again

    print('Set state')
    env._set_state(sampled_state)
    next_state = env.true_dynamic(sampled_state, action)
    print("Next state:", next_state)
    env._get_obs()