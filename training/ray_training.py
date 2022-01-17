import json
import pkg_resources

from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.env import PettingZooEnv
from ray.rllib.agents import ppo

from pz_vehicle import vehicle_env

if __name__ == '__main__':
    config_data = json.load(open(pkg_resources.resource_filename(__name__, "./config.json")))
    register_env('vehicle', lambda config: PettingZooEnv(vehicle_env.env(config_data, 0)))
    tf_config = ppo.DEFAULT_CONFIG.copy()
    trainer = ppo.PPOTrainer(config=tf_config, env="vehicle")
    for i in range(10000):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result["info.learner.default_policy.learner_stats"]))

        if i % 1000 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
        if i % 10 == 0:
            print('progress: ', i / 100, '%')
    pass
