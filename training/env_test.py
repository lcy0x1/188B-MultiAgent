import json

import numpy as np
import pkg_resources

from pz_vehicle import vehicle_env

if __name__ == '__main__':
    config_data = json.load(open(pkg_resources.resource_filename(__name__, "./config.json")))
    env = vehicle_env.env(config_data, 0)
    print(config_data["vehicle"])
    env.reset()
    agents = env.agents
    print(agents)
    print('Agent 1 state: ', env.observe(agents[0]))
    print('Agent 2 state: ', env.observe(agents[1]))
    env.step(np.array([1, 1, 0.5]))
    env.step(np.array([1, 1, 0.5]))
    print('Agent 1 state: ', env.observe(agents[0]))
    print('Agent 2 state: ', env.observe(agents[1]))
    print('Agent 1 info: ', env.infos[agents[0]])
    print('Agent 2 info: ', env.infos[agents[1]])
    print('Agent 1 reward: ', env.rewards[agents[0]])
    print('Agent 2 reward: ', env.rewards[agents[1]])
    env.step(np.array([1, 2, 0.5]))
    env.step(np.array([2, 1, 0.5]))
    print('Agent 1 state: ', env.observe(agents[0]))
    print('Agent 2 state: ', env.observe(agents[1]))
    print('Agent 1 info: ', env.infos[agents[0]])
    print('Agent 2 info: ', env.infos[agents[1]])
    print('Agent 1 reward: ', env.rewards[agents[0]])
    print('Agent 2 reward: ', env.rewards[agents[1]])
