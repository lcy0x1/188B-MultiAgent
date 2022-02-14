import math
import statistics

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import gym_vehicle
import gym_symmetric
import csv


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def plot(env, path, fac):
    n = 10
    m = 100
    mean_list = []
    stdev_list = []
    for i in range(100):
        model = PPO.load(f"./data_{path}_n4v4_set1/{i + 1}")
        model.set_env(env)

        list_sums = []
        for trial in range(n):
            obs = env.reset()
            sums = 0
            for _ in range(m * fac):
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                sums = sums + rewards
            sums = sums / m
            list_sums.append(sums)
        print(f"DeepRL {i + 1}: average return: ", statistics.mean(list_sums), ", stdev = ",
              statistics.stdev(list_sums))
        mean_list.append(statistics.mean(list_sums))
        stdev_list.append(statistics.stdev(list_sums))
    with open(f'data_{path}_n4v4_stats.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(mean_list)
        tsv_writer.writerow(stdev_list)


if __name__ == "__main__":
    plot(make_env("vehicle-v0", 12345)(), 'cls', 1)
    plot(make_env("symmetric-v0", 12345)(), 'sym', 4)
