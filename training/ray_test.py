import json

from ray.rllib.utils import tf_utils
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents import ppo

from gym_vehicle import envs

if __name__ == '__main__':

    register_env('gym_vehicle', lambda config: envs.VehicleEnv())
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
            log_list.append({"step": i, "status": status, "checkpoint": checkpoint})
            out_file = open("log.json", "w")
            json.dump(log_list, out_file)
            out_file.close()
