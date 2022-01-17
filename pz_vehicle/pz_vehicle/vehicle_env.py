import math
import numpy as np
from gym.spaces import MultiDiscrete, Box
from gym.utils import seeding
from numpy.random import RandomState
from pettingzoo.utils import AECEnv, wrappers
from typing import List, Optional


class Config:

    def __init__(self, json, seed=0):
        self.node = json["node"]
        self.vehicle = json["vehicle"]
        self.max_queue = json["max_queue"]
        self.operation_cost = json["operation_cost"]
        self.waiting_penalty = json["waiting_penalty"]
        self.poisson_cap = json["poisson_cap"]
        self.poisson_param = json["poisson_param"]
        self.overflow = json["overflow"]
        self.seed = seed

    def gen_distribution(self, random_func):
        arr = random_func.random_sample(self.node)
        ans = [0 for _ in range(self.node)]
        rem = self.vehicle
        for i in range(self.node):
            arr[i] *= self.vehicle
            ans[i] = math.floor(arr[i])
            arr[i] -= ans[i]
            rem -= ans[i]
        for i in range(rem):
            rand = random_func.random_sample() * rem
            for j in range(self.node):
                if rand < arr[j]:
                    ans[j] += 1
                    break
                rand -= arr[j]
        return ans


class AgentAction:

    def __init__(self, arr, self_index, node, vehicles, random_func: RandomState):
        self.motion = [0 for _ in range(node)]
        self.price = [0.0 for _ in range(node)]
        ind = 0
        tmp = [0 for _ in range(node)]

        # find the total weights
        # assign raw weights to tmp
        raw_sum = 0
        for i in range(node):
            tmp[i] = arr[ind]
            raw_sum = raw_sum + arr[ind]
            ind = ind + 1
        # prevent div-0 error
        raw_sum = max(1e-5, raw_sum)

        # calculate the number to randomly distribute
        rem: int = vehicles
        for i in range(node):
            tmp[i] = vehicles * tmp[i] / raw_sum
            rem = rem - math.floor(tmp[i])
            self.motion[i] = math.floor(tmp[i])
            tmp[i] -= self.motion[i]

        # Randomly distribute the rest of the vehicles
        # The tmp decimals sums up to the remaining number of vehicles
        # Generate the same number of random number in [0,1), add all up
        # and find matching place in the spectrum from tmp
        for i in range(rem):
            rand = random_func.random_sample() * rem
            for j in range(node):
                if rand < tmp[j]:
                    self.motion[j] += 1
                    break
                rand -= tmp[j]

        # skip itself for price
        for i in range(node):
            if i == self_index:
                continue
            self.price[i] = arr[ind]
            ind = ind + 1


def env(config, seed=0):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    environment = VehicleEnv(Config(config, seed))
    environment = wrappers.CaptureStdoutWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


class VehicleEnv(AECEnv):
    metadata = {
        "name": 'vehicle_v0',
        "render.modes": ["human"]
    }

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.possible_agents = [f"node_{i}" for i in range(config.node)]
        self.observation_spaces = {
            agent: MultiDiscrete([config.vehicle + 1] + [config.max_queue + 1 for _ in range(config.node - 1)])
            for agent in self.possible_agents
        }
        self.action_spaces = {agent: Box(0, 1, (config.node * 2 - 1,)) for agent in self.possible_agents}
        self._random_func, _ = seeding.np_random(config.seed)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {"cost": 0, "waiting_penalty": 0, "overflow": 0, "reward": 0} for agent in self.agents}
        self._states = [[] for i in range(self.config.node)]
        self.states = {agent: [] for agent in self.agents}
        self.observations = self.states
        self.num_moves = 0
        self._agent_index = 0
        self.agent_selection = self.agents[self._agent_index]
        self._action_cache: List[Optional[AgentAction]] = [None for _ in range(self.config.node)]

    def reset(self):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.num_moves = 0
        self._agent_index = 0
        self.agent_selection = self.agents[self._agent_index]

        init_distribution = self.config.gen_distribution(self._random_func)
        self._states = [[init_distribution[i]] + [0 for _ in range(self.config.node - 1)]
                        for i in range(self.config.node)]

        self.infos = {agent: {} for agent in self.agents}
        self.states = {self.agents[i]: self._states[i] for i in range(self.config.node)}
        self.observations = self.states
        self._action_cache = [None for _ in range(self.config.node)]

    def step(self, action):
        """
                step(action) takes in an action for the current agent (specified by
                agent_selection) and needs to update
                - rewards
                - _cumulative_rewards (accumulating the rewards)
                - dones
                - infos
                - agent_selection (to the next agent)
                And any internal state used by observe() or render()
                """
        act = AgentAction(action, self._agent_index, self.config.node, self.observations[self.agent_selection][0],
                          self._random_func)

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self._action_cache[self._agent_index] = act

        self._agent_index += 1
        if self._agent_index == self.config.node:
            self._agent_index = 0
            self._env_step()
        self.agent_selection = self.agents[self._agent_index]

    def observe(self, agent):
        return np.array(self.observations[agent])

    def render(self, mode='human'):
        pass

    def state(self):
        return self.states

    def observation_space(self, agent):
        # vehicle (1) goes first, queue (n-1) goes second
        return self.observation_spaces[agent]

    def action_space(self, agent):
        # vehicle action (n) goes first, price (n-1) goes second
        return self.action_spaces[agent]

    def _env_step(self):
        self.num_moves += 1
        for i in range(self.config.node):
            act = self._action_cache[i]
            cost = 0
            wait_pen = 0
            overf = 0
            reward = 0
            for j in range(self.config.node):
                if i == j:
                    continue
                ind = j
                if j < i:
                    ind += 1
                self._states[i][0] -= act.motion[j]
                self._states[j][0] += act.motion[j]
                self._states[i][ind] -= act.motion[j]
                if self._states[i][ind] < 0:
                    self._states[i][ind] = 0
                cost += self.config.operation_cost * act.motion[j]
                wait_pen += self.config.waiting_penalty * self._states[i][ind]

                request = min(self.config.poisson_cap,
                              self._random_func.poisson(self.config.poisson_param * (1 - act.price[j])))
                act_req = min(request, self.config.max_queue - self._states[i][ind])
                overf += (request - act_req) * self.config.overflow
                self._states[i][ind] += act_req
                reward += act_req * act.price[j]
            total = reward - cost - wait_pen - overf
            self.rewards[self.agents[i]] = total
            self._cumulative_rewards[self.agents[i]] += total
            self.infos[self.agents[i]] = {"cost": cost, "waiting_penalty": wait_pen, "overflow": overf,
                                          "reward": reward, "action.motion": act.motion, "action.price": act.price}
