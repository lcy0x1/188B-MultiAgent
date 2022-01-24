import json
import pkg_resources
from datetime import datetime

from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents import ppo

from gym_vehicle import envs

if __name__ == '__main__':
    config_data = json.load(open(pkg_resources.resource_filename(__name__, "./config.json")))
    register_env('gym_vehicle', lambda config: envs.VehicleEnv(config_data, 0))
    tf_config = ppo.DEFAULT_CONFIG.copy()
    trainer = ppo.PPOTrainer(config=tf_config, env="gym_vehicle")
    log_file = open("./gym_log.json", "r")
    logs = json.load(log_file)
    trainer.restore(logs[-1]["checkpoint"])
    env = envs.VehicleEnv(config_data, 0)
    env.reset()
    obs = env.to_observation()
    total_reward = 0
    n = 10000
    for i in range(n):
        action = trainer.compute_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    print(total_reward / n)
