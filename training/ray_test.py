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
    env = envs.VehicleEnv(config_data, 0)
    log_list = []
    for i in range(100):
        # Perform one iteration of training the policy with PPO
        result = None
        for _ in range(100):
            result = trainer.train()
        print(pretty_print(result))
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

        obs = env.reset()
        total_reward = 0
        n = 1000
        print(env.action_space)
        for _ in range(n):
            action = trainer.compute_single_action(obs, clip_action=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print(total_reward / n)

        status = result["info"]["learner"]["default_policy"]["learner_stats"]
        status = {"policy_loss": str(status["policy_loss"]),
                  "value_function_loss": str(status["vf_loss"]),
                  "total_loss": str(status["total_loss"])}
        log_list.append({"step": i, "status": status, "reward": total_reward / n, "checkpoint": checkpoint,
                         "timestamp": str(datetime.fromtimestamp(result["timestamp"]))})
        out_file = open("./gym_log.json", "w")
        json.dump(log_list, out_file)
        out_file.close()
