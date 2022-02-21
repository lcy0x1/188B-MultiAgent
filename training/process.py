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


def plot(env, path, fac, tr_begin, tr_end):
    n = 100
    m = 1000
    matrix = []
    for i in range(tr_end - tr_begin):
        model = PPO.load(f"./data_{path}_set1/{i + 1 + tr_begin}")
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
        print(f"DeepRL {i + 1 + tr_begin}: average return: ", statistics.mean(list_sums), ", stdev = ",
              statistics.stdev(list_sums))
        matrix.append(list_sums)
    with open(f'data_{path}_{tr_begin}_{tr_end}_stats.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerows(matrix)


if __name__ == "__main__":
    plot(make_env("symmetric-v0", 12345)(), 'sym_n8v8', 8, 140, 170)
