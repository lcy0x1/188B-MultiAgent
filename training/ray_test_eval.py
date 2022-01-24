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
    tf_config["explore"] = False
    trainer = ppo.PPOTrainer(config=tf_config, env="gym_vehicle")
    trainer.restore("../../ray_results/PPO_gym_vehicle_2022-01-18_00-25-13gc7rendc/checkpoint_000501/checkpoint-501")
    env = envs.VehicleEnv(config_data, 0)
    obs = env.reset()
    total_reward = 0
    n = 10000
    for i in range(n):
        print("step: ", i)
        print(obs)
        action = trainer.compute_single_action(obs)
        print(action)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    print(total_reward / n)
