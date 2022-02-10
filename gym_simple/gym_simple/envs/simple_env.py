import gym
import math
import pkg_resources
from gym import error, spaces, utils
from gym.utils import seeding
import json


class SimpleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config=None, seed=0):
        self.random = None
        self.n = 8
        self.observation_space = spaces.MultiDiscrete([self.n + 1, self.n + 1])
        self.action_space = spaces.Box(0, 1, (2,))
        self._numbers = [self.n, 0]
        self.seed(seed)

    def seed(self, seed=None):
        self.random, _ = seeding.np_random(seed)

    def step(self, act):
        val_tmp = self._numbers[0] * act[1] / max(1e-5, sum(act))
        val = math.floor(val_tmp)
        if self.random.rand(1) < val_tmp - val:
            val += 1
        self._numbers[0] -= val
        self._numbers[1] += val
        reward = val
        if self._numbers[1] > self._numbers[0]:
            reward -= 10
        if self._numbers[1] > 0:
            self._numbers[0] += 1
            self._numbers[1] -= 1
        return self.to_observation(), reward, False, {}

    def reset(self):
        self._numbers = [self.n, 0]
        return self.to_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def to_observation(self):
        return self._numbers
