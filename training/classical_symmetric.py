import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import gym_symmetric


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        print("initial state: ", env.reset())
        print("Action Space: ", env.observation_space)
        print("Observation Space: ", env.action_space)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    env_id = "symmetric-v1"
    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    model = PPO('MlpPolicy', env, verbose=1)

    # model = PPO.load("./data/1mil")
    # model.set_env(env)

    sum_list = []

    for i in range(200):
        model.learn(total_timesteps=1_000_000)
        model.save(f"./data_sym_n8v8_reg_set1/{i + 1}")
        accu = 0
        for _ in range(100):
            sums = 0
            j = 0
            obs = env.reset()
            for _ in range(800):
                j += 1
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                sums = sums + rewards
            accu += sums / j * 8
        sum_list.append(accu/100)
        print("average return: ", accu)
    print("average return: ", sum_list)
