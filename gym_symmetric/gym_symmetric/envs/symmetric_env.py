from typing import List, Optional

import gym
import math
import pkg_resources
from gym import spaces
from gym.utils import seeding
import json
import sys

from numpy.random.mtrand import RandomState


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
        self.vehicles = [0 for _ in range(self.node)]
        self.queue = [[0 for _ in range(self.node)] for _ in range(self.node)]

        # Attempt at edge initialization
        # Edge matrix: self.edge(0) = 1->2 , self.edge(1) = 2->1     for 2 node case (2 edges)
        # n nodes: self.edge(0) = 1->2 , 1->3 , ... 1->n , 2->1 , 2->3, ... 2->n , ... n->n-2 , n->n-1  (? edges)
        self.edge_list = self.config["edge_lengths"]
        self.edge_matrix = [[0 for _ in range(self.node)] for _ in range(self.node)]
        self.bounds = [0 for _ in range(self.node)]
        self.fill_edge_matrix()
        self.max_edge = max(self.bounds)

        self.current_index = 0
        self.action_cache: List[Optional[VehicleAction]] = [None for _ in range(self.node)]
        self.over = 0
        self.random: Optional[RandomState] = None
        self.observation_space = spaces.MultiDiscrete(
            [self.vehicle + 1 for _ in range(sum(self.bounds))] +  # vehicles
            [self.queue_size + 1 for _ in range(self.node)] +  # queue
            [self.queue_size * (self.node - 1) + 1 for _ in range(self.node)] +  # queue at other nodes
            [self.max_edge + 1 for _ in range(self.node)] +
            [self.node])  # state
        self.action_space = spaces.Box(0, 1, (self.node * 2,))

        # Stores number of vehicles at mini node between i and j
        self.mini_vehicles = [[[0 for _ in range(self.edge_matrix[i][j] - 1)]
                               for j in range(self.node)] for i in range(self.node)]

        self.seed(seed)

    def seed(self, seed=None):
        self.random, _ = seeding.np_random(seed)

    def fill_edge_matrix(self):
        edge_num = len(self.edge_list)
        if (self.node * (self.node - 1)) != edge_num:
            print("Incorrect edge_lengths parameter. Total nodes and edges do not match!")
            sys.exit()
        # Creating 2D matrix for easier access
        tmp = 0
        for i in range(self.node):
            for j in range(self.node):
                self.edge_matrix[i][j] = self.edge_list[tmp]
                self.bounds[j] = max(self.bounds[j], self.edge_matrix[i][j])
                tmp += 1
                if i == j:
                    continue
                if self.edge_matrix[i][j] < 1:
                    print("Error! Edge length too short (minimum length 1).")
                    sys.exit()
                if self.edge_matrix[i][j] % 1 != 0:
                    print("Error! Edge length must be integer value.")
                    sys.exit()

    def step(self, act):
        action = VehicleAction(self, self.current_index, act)
        self.action_cache[self.current_index] = action
        self.current_index += 1
        if self.current_index == self.node:
            self.current_index = 0
            op_cost = 0
            wait_pen = 0
            overf = 0
            rew = 0

            # Move cars in mini-nodes ahead
            for i in range(self.node):
                for j in range(self.node):
                    if i == j:
                        continue
                    # Sweeping BACKWARDS to avoid pushing vehicles multiple times in same time step
                    for m in range(self.edge_matrix[i][j] - 1):
                        if m == 0:
                            # Stop tracking mini-node behavior and push cars to main node
                            self.vehicles[j] += self.mini_vehicles[i][j][m]
                        else:
                            # Vehicles still in mini nodes (traveling)
                            # Shifting vehicles further along path
                            self.mini_vehicles[i][j][m - 1] = self.mini_vehicles[i][j][m]
                        op_cost += self.mini_vehicles[i][j][m] * self.operating_cost
                        self.mini_vehicles[i][j][m] = 0

            for i in range(self.node):
                for j in range(self.node):
                    if i == j:
                        continue
                    veh_motion = self.action_cache[i].motion[j]
                    # Statement to feed to mini-nodes
                    # Only feed to mini nodes if required (edge length > 1)   ->   Feed to first mini-node
                    if self.edge_matrix[i][j] > 1:
                        # for distance 2, it feeds to the 1st mininode (index 0)
                        # for distance 5, it feeds to the 4th mininode (index 3)
                        self.mini_vehicles[i][j][self.edge_matrix[i][j] - 2] += veh_motion
                    else:
                        # Cars arriving at node j (for length 1 case)
                        self.vehicles[j] += veh_motion
                    self.vehicles[i] -= veh_motion
                    self.queue[i][j] = max(0, self.queue[i][j] - veh_motion)
                    op_cost += veh_motion * self.operating_cost
                    wait_pen += self.queue[i][j] * self.waiting_penalty
                    price = self.action_cache[i].price[j]
                    edge_len = self.edge_matrix[i][j]
                    freq = self.poisson_param * (1 - price)
                    request = min(self.poisson_cap, self.random.poisson(freq))
                    act_req = request
                    if self.queue[i][j] + act_req > self.queue_size:
                        act_req = 0
                    overf += (request - act_req) * self.overflow
                    self.queue[i][j] = self.queue[i][j] + act_req
                    rew += act_req * price * edge_len
            debuf_info = {'loss': rew, 'operating_cost': op_cost, 'wait_penalty': wait_pen, 'overflow': overf}
            return self.to_observation(), rew - op_cost - wait_pen - overf, False, debuf_info
        return self.to_observation(), 0, False, {}

    def reset(self):
        # Reset queue, vehicles at nodes AND in travel
        for i in range(self.node):
            self.vehicles[i] = 0
            for j in range(self.node):
                self.queue[i][j] = 0
                for k in range(self.edge_matrix[i][j] - 1):
                    self.mini_vehicles[i][j][k] = 0

        for i in range(self.vehicle):
            pos = self.random.randint(0, self.node)
            self.vehicles[pos] = self.vehicles[pos] + 1
        self.over = 0
        self.current_index = 0
        return self.to_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def to_observation(self):
        arr = [0 for _ in range(self.node * 3 + sum(self.bounds) + 1)]
        ind = 0
        for i in range(self.node):
            arr[ind] = self.vehicles[i]
            ind += 1
        for j in range(self.node):
            sums = [0 for _ in range(self.bounds[j] - 1)]
            for i in range(self.node):
                for k in range(self.edge_matrix[i][j] - 1):
                    sums[k] += self.mini_vehicles[i][j][k]
            for k in range(self.bounds[j] - 1):
                arr[ind] += sums[k]
                ind += 1
        for i in range(self.node):
            arr[ind] = self.queue[self.current_index][i]
            ind += 1
        for i in range(self.node):
            arr[ind] = sum(self.queue[i])
            ind += 1
        for i in range(self.node):
            arr[ind] = self.edge_matrix[self.current_index][i]
            ind += 1
        arr[ind] = self.current_index
        return arr
