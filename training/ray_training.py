import json
import pkg_resources

from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.env import PettingZooEnv
from ray.rllib.agents import ppo

from pz_vehicle import vehicle_env


if __name__ == '__main__':
    config_data = json.load(open(pkg_resources.resource_filename(__name__, "./config.json")))
    register_env('vehicle', lambda config: PettingZooEnv(vehicle_env.env(config_data)))

    trainer = ppo.PPOTrainer(env="vehicle")
    trainer.setup()
    result = trainer.train()
    print(pretty_print(result))
    pass
