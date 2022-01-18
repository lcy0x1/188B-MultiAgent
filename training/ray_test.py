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
    log_list = []
    for i in range(10000):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
            status = result["info"]["learner"]["default_policy"]["learner_stats"]
            status = {"policy_loss": str(status["policy_loss"]),
                      "value_function_loss": str(status["vf_loss"]),
                      "total_loss": str(status["total_loss"])}
            log_list.append({"step": i, "status": status, "checkpoint": checkpoint,
                             "timestamp": str(datetime.fromtimestamp(result["timestamp"]))})
            out_file = open("./gym_log.json", "w")
            json.dump(log_list, out_file)
            out_file.close()
