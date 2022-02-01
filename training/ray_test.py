import json
import pkg_resources
from datetime import datetime

from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents import ppo

from gym_vehicle import envs

if __name__ == '__main__':
    ppo_config = {
        # Environment (RLlib understands openAI gym registered strings).
        "env": "gym_vehicle",
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "num_workers": 2,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "tf",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        }
    }
    config_data = json.load(open(pkg_resources.resource_filename(__name__, "./config.json")))
    register_env('gym_vehicle', lambda config: envs.VehicleEnv(config_data, 0))
    trainer = ppo.PPOTrainer(config=ppo_config)
    env = envs.VehicleEnv(config_data, 0)
    log_list = []
    for i in range(100):
        # Perform one iteration of training the policy with PPO
        result = None
        result = trainer.train()
        print(pretty_print(result))
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

        obs = env.reset()
        total_reward = 0
        n = 1000
        print(env.action_space)
        for _ in range(n):
            action = trainer.compute_single_action(obs, explore=False, clip_action=True)
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
