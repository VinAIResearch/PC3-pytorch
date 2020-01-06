import math

import numpy as np

import gym
from gym import spaces
from PIL import Image
class MountainCarMDP(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    action_dim = 1
    min_action = -1.0
    max_action = 1.0
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
    goal_velocity = 0.
    goal_reward = 1
    power = 0.0015

    def __init__(self, width=60, height=40, noise=0.0):
        self.position_range = [self.min_position, self.max_position]
        self.speed_range = [-self.max_speed, self.max_speed]
        self.action_range = np.array([self.min_action, self.max_action])
        self.viewer = None

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])
        self.action_space = spaces.Box(low=self.action_range[0], high=self.action_range[1],
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        self.width = width
        self.height = height
        self.noise = noise

    def take_step(self, s, u):
        position = s[0]
        velocity = s[1]

        force = np.squeeze(np.clip(u, self.action_range[0], self.action_range[1]))

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0

        return np.array([position, velocity])

    def transition_function(self, s, u):  # compute next state and add noise
        s_next = self.take_step(s, u)
        # add noise
        s_next += self.noise * np.random.rand(*s_next.shape)
        return s_next

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def render_obs(self, s, mode='rgb_array'):
        screen_width = 60
        screen_height = 40

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=6
        carheight=3

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(1)
            self.viewer.add_geom(self.track)

            clearance = 1

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            # frontwheel = rendering.make_circle(carheight/2.5)
            # frontwheel.set_color(.5, .5, .5)
            # frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            # frontwheel.add_attr(self.cartrans)
            # self.viewer.add_geom(frontwheel)
            # backwheel = rendering.make_circle(carheight/2.5)
            # backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            # backwheel.add_attr(self.cartrans)
            # backwheel.set_color(.5, .5, .5)
            # self.viewer.add_geom(backwheel)

        pos = s[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        arr = self.viewer.render(return_rgb_array = mode == 'rgb_array')
        img = Image.fromarray(arr)
        img = img.convert('L').resize((self.height,
                                   self.width))

        return np.expand_dims(np.asarray(img).squeeze() / 255.0, axis=-1)

    def is_goal(self, s):
        """Check if the pendulum is in goal region"""
        position = s[0]
        velocity = s[1]
        return position >= self.goal_position and velocity >= self.goal_velocity

    def is_fail(self, s):
        return False

    def reward_function(self, s):
        """Reward function."""
        return int(self.is_goal(s)) * self.goal_reward

    def sample_random_state(self):
        position = np.random.uniform(self.position_range[0], self.position_range[1])
        velocity = np.random.uniform(self.speed_range[0],
                                       self.speed_range[1])
        true_state = np.array([position, velocity])
        return true_state

    def sample_random_action(self):
        """Sample a random action from action range."""
        return np.array(
            [np.random.uniform(self.action_range[0], self.action_range[1])])

    def sample_extreme_action(self):
        """Sample a random extreme action from action range."""
        return np.array(
            [np.random.choice([self.action_range[0], self.action_range[1]])])

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None