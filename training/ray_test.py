import json
import pkg_resources
from datetime import datetime

from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents import ddpg

from gym_vehicle import envs

if __name__ == '__main__':
    ddpg_config = {
        "env": "gym_vehicle",
        "framework": "tf",
        "horizon": 1000,
        "soft_horizon": False
    }
    config_data = json.load(open(pkg_resources.resource_filename(__name__, "./config.json")))
    register_env('gym_vehicle', lambda config: envs.VehicleEnv(config_data, 0))
    trainer = ddpg.DDPGTrainer(config=ddpg_config)
    env = envs.VehicleEnv(config_data, 0)
    log_list = []
    for i in range(100):
        # Perform one iteration of training the policy with PPO
        result = None
        for _ in range(10):
            result = trainer.train()
        print(pretty_print(result))
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

        obs = env.reset()
        total_reward = 0
        overf = False
        n = 1000
        print(env.action_space)
        for _ in range(n):
            action = trainer.compute_single_action(obs, explore=False, clip_action=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if reward < -4:
                overf = True
        print(total_reward / n, overf)

        status = result["info"]["learner"]["default_policy"]["learner_stats"]
        status = {"policy_loss": str(status["policy_loss"]),
                  "value_function_loss": str(status["vf_loss"]),
                  "total_loss": str(status["total_loss"])}
        log_list.append({"step": i, "status": status, "reward": total_reward / n, "checkpoint": checkpoint,
                         "timestamp": str(datetime.fromtimestamp(result["timestamp"]))})
        out_file = open("./gym_log.json", "w")
        json.dump(log_list, out_file)
        out_file.close()
