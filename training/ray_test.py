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
        accu = 0
        overf = False
        for _ in range(100):
            total_reward = 0
            j = 0
            env.reset()
            for _ in range(100):
                j += 1
                action = trainer.compute_single_action(obs, explore=False, clip_action=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    overf = True
                    break
            accu += total_reward / j
        print(accu/100, overf)

        log_list.append({"step": i, "reward": accu/100, "checkpoint": checkpoint,
                         "timestamp": str(datetime.fromtimestamp(result["timestamp"]))})
        out_file = open("./gym_log.json", "w")
        json.dump(log_list, out_file)
        out_file.close()
