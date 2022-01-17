import json
import pkg_resources
import torch

from ray.rllib.utils import tf_utils

from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.env import PettingZooEnv
from ray.rllib.agents import ppo

from pz_vehicle import vehicle_env

if __name__ == '__main__':
    config_data = json.load(open(pkg_resources.resource_filename(__name__, "./config.json")))
    register_env('vehicle', lambda config: PettingZooEnv(vehicle_env.env(config_data, 0)))
    tf_config = ppo.DEFAULT_CONFIG.copy()
    tf_num_gpus = len(tf_utils.get_gpu_devices())
    tc_num_gpus = torch.cuda.device_count()
    print("Number of GPU detected - TensorFlow: ", tf_num_gpus, " Torch: ", tc_num_gpus)
    if tc_num_gpus > tf_num_gpus:
        tf_config["framework"] = "torch"
        tf_config["num_gpus"] = tc_num_gpus
    else:
        tf_config["framework"] = "tf"
        tf_config["num_gpus"] = tf_num_gpus
    trainer = ppo.PPOTrainer(config=tf_config, env="vehicle")
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
            log_list.append({"step": i, "status": status, "checkpoint": checkpoint})
            out_file = open("log.json", "w")
            json.dump(log_list, out_file)
            out_file.close()
