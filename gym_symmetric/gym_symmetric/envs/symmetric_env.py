from typing import List, Optional

import gym
import math
import pkg_resources
from gym import spaces
from gym.utils import seeding
import json


class VehicleAction:

    def __init__(self, env, i, arr):
        self.motion = [0 for _ in range(env.node)]
        self.price = [0.0 for _ in range(env.node)]
        ind = 0
        tmp = [0 for _ in range(env.node)]
        rsum = 0
        for j in range(env.node):
            tmp[j] = min(1, max(0, arr[ind]))
            rsum = rsum + tmp[j]
            ind = ind + 1
        rsum = max(1e-5, rsum)
        rem: int = env.vehicles[i]
        for j in range(env.node):
            tmp[j] = env.vehicles[i] * tmp[j] / rsum
            rem = rem - math.floor(tmp[j])
        random = env.random.rand(1)
        rem = rem - 1
        for j in range(env.node):
            mrem = tmp[j] - math.floor(tmp[j])
            if (random > 0) and (random < mrem):
                self.motion[j] = math.floor(tmp[j]) + 1
                if rem > 0:
                    random = random + env.random.rand(1)
                    rem = rem - 1
            else:
                self.motion[j] = math.floor(tmp[j])
            random = random - mrem
        for j in range(env.node):
            if i != j:
                self.price[j] = min(1, max(0, arr[ind]))
            ind = ind + 1


class VehicleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config=None, seed=0):
        if config is None:
            config = json.load(open(pkg_resources.resource_filename(__name__, "./config.json")))
        self.config = config
        self.node = self.config["node"]
        self.vehicle = self.config["vehicle"]
        self.poisson_param = self.config["poisson_param"]
        self.operating_cost = self.config["operation_cost"]
        self.waiting_penalty = self.config["waiting_penalty"]
        self.queue_size = self.config["max_queue"]
        self.overflow = self.config["overflow"]
        self.poisson_cap = self.config["poisson_cap"]
        self.length_regulator = self.config["length_regulator"]
        self.vehicles = [0 for _ in range(self.node)]
        self.queue = [[0 for _ in range(self.node)] for _ in range(self.node)]
        self.current_index = 0
        self.action_cache: List[Optional[VehicleAction]] = [None for _ in range(self.node)]
        self.over = 0
        self.random = None
        self.observation_space = spaces.MultiDiscrete(
            [self.vehicle + 1 for _ in range(self.node)] +  # vehicles
            [self.queue_size + 1 for _ in range(self.node - 1)] +  # queue
            [self.queue_size * (self.node - 1) + 1 for _ in range(self.node - 1)] +  # queue at other nodes
            [self.node])  # state
        self.action_space = spaces.Box(0, 1, (self.node * 2,))
        self.seed(seed)

    def seed(self, seed=None):
        self.random, _ = seeding.np_random(seed)

    def step(self, act):
        action = VehicleAction(self, self.current_index, act)
        self.action_cache[self.current_index] = action
        self.current_index += 1
        len_pen = abs(sum(act[0:self.node]) - 1) * self.length_regulator
        if self.current_index == self.node:
            self.current_index = 0
            op_cost = 0
            wait_pen = 0
            overf = 0
            rew = 0
            for i in range(self.node):
                for j in range(self.node):
                    if i == j:
                        continue
                    veh_motion = self.action_cache[i].motion[j]
                    self.vehicles[i] -= veh_motion
                    self.vehicles[j] += veh_motion
                    self.queue[i][j] = max(0, self.queue[i][j] - veh_motion)
                    op_cost += veh_motion * self.operating_cost
                    wait_pen += self.queue[i][j] * self.waiting_penalty
                    price = self.action_cache[i].price[j]
                    request = min(self.poisson_cap, self.random.poisson(self.poisson_param * (1 - price)))
                    act_req = request
                    if self.queue[i][j] + act_req > self.queue_size:
                        act_req = 0
                    overf += (request - act_req) * self.overflow
                    self.queue[i][j] = self.queue[i][j] + act_req
                    rew += act_req * self.action_cache[i].price[j]
            debuf_info = {'loss': rew, 'operating_cost': op_cost, 'wait_penalty': wait_pen, 'overflow': overf}
            return self.to_observation(), rew - op_cost - wait_pen - overf - len_pen, False, debuf_info
        return self.to_observation(), -len_pen, False, {}

    def reset(self):
        for i in range(self.node):
            self.vehicles[i] = 0
            for j in range(self.node):
                self.queue[i][j] = 0
        for i in range(self.vehicle):
            pos = self.random.randint(0, self.node)
            self.vehicles[pos] = self.vehicles[pos] + 1
        self.over = 0
        return self.to_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def to_observation(self):
        arr = [0 for _ in range(self.node * 3 - 1)]
        for i in range(self.node):
            j = (self.current_index + i) % self.node
            arr[i] = self.vehicles[j]
            if i > 0:
                arr[self.node + i - 1] = self.queue[self.current_index][j]
                arr[self.node * 2 + i - 2] = sum(self.queue[j])
        arr[self.node * 3 - 1] = self.current_index
        return arr
