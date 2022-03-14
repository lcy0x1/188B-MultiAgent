import csv
import json
import statistics
import sys

import gym
import numpy as np
import pkg_resources
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import gym_vehicle
import gym_symmetric
from torch import nn

if __name__ == "__main__":
    id = sys.argv[1]
    layer_n = int(sys.argv[2])
    layer_l = int(sys.argv[3])
    mil_steps = int(sys.argv[4])
    eval_n = int(sys.argv[5])
    eval_m = int(sys.argv[6])
    eval_k = int(sys.argv[7])

    n_env = 64  # Number of processes to use
    # Create the vectorized environment
    env = make_vec_env(id, n_env)

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    layers = [layer_n for _ in range(layer_l)]
    policy_kwargs = {
        "net_arch": [{"vi": layers, "vf": layers}],
        "activation_fn": nn.ReLU
    }
    network_type = '-'.join(list(map(str, layers)))
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, n_steps=128, verbose=0)

    # model = PPO.load("./data/1mil")
    # model.set_env(env)

    for i in range(mil_steps):
        for _ in range(122):
            model.learn(total_timesteps=128 * 64)
        model.save(f"./data/n8v16/{network_type}_relu/{id}/{i + 1}")
        accu = 0

        list_sums = []
        for _ in range(eval_n):
            sums = 0
            j = 0
            obs = env.reset()
            for _ in range(eval_m * eval_k):
                j += 1
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                sums = sums + rewards
            list_sums.extend((sums / eval_m).tolist())
        with open(f"./data/n8v16/{network_type}_relu/{id}.tsv", 'a') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(list_sums)
        print(f"{network_type}/{id}/{i + 1}: average return: ", statistics.mean(list_sums), ", stdev = ",
              statistics.stdev(list_sums))
