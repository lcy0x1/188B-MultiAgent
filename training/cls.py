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


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        print("initial state: ", env.reset())
        print("Observation Space: ", env.observation_space)
        print("Action Space: ", env.action_space)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    id = sys.argv[1]
    layer_n = int(sys.argv[2])
    layer_l = int(sys.argv[3])
    mil_steps = int(sys.argv[4])
    eval_n = int(sys.argv[5])
    eval_m = int(sys.argv[6])
    eval_k = int(sys.argv[7])
    lrate = int(sys.argv[8])

    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = DummyVecEnv([make_env(id, i) for i in range(num_cpu)])

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
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0,
                gamma=0.99 ** (1 / eval_k), gae_lambda=0.95 ** (1 / eval_k),
                n_steps=256 * eval_k, learning_rate=lrate * 0.0003)

    # model = PPO.load("./data/1mil")
    # model.set_env(env)

    nid = "avg_" + id
    dire = f"./data/n9v150ns/{network_type}-lr{lrate}/"

    for i in range(mil_steps):
        model.learn(total_timesteps=1_000_000)
        model.save(dire + f"{nid}/{i + 1}")
        accu = 0

        list_reward = []
        list_lf = []
        list_q = []
        list_p = []
        for _ in range(eval_n):
            sum_reward = np.zeros((8,))
            sum_queue = np.zeros((8,))
            sum_price = np.zeros((8,))
            j = 0
            obs = env.reset()
            for _ in range(eval_m * eval_k):
                j += 1
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                if j % eval_k == 0:
                    sum_reward += np.array([v['reward'] for v in info])
                    sum_queue += np.array([v['avg_queue'] for v in info])
                    sum_price += np.array([v['avg_price'] for v in info])
            list_reward.extend((sum_reward / eval_m).tolist())
            list_q.extend((sum_queue / eval_m).tolist())
            list_p.extend((sum_price / eval_m).tolist())
        with open(dire + f"{nid}stats/reward.tsv", 'a') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(list_reward)
        with open(dire + f"{nid}stats/queue.tsv", 'a') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(list_q)
        with open(dire + f"{nid}stats/price.tsv", 'a') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(list_p)
        print(f"{network_type}/{nid}/{i + 1}: average return: ", statistics.mean(list_reward), ", stdev = ", statistics.stdev(list_reward))
        print(f"{network_type}/{nid}/{i + 1}: average queue: ", statistics.mean(list_q), ", stdev = ", statistics.stdev(list_q))
        print(f"{network_type}/{nid}/{i + 1}: average price: ", statistics.mean(list_p), ", stdev = ", statistics.stdev(list_p))
