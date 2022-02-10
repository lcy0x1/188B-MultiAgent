import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import gym_vehicle
import gym_simple


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
    env_id = "simple-v0"
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

    sums = 0
    obs = env.reset()
    _, rewards, _, _ = env.step(np.array([[0.5, 0.5] for _ in range(8)]))
    sums = sums + rewards
    for _ in range(999):
        _, rewards, _, _ = env.step(np.array([[0.8, 0.2] for _ in range(8)]))
        sums = sums + rewards
    print(sums / 1000)

    for i in range(100):
        model.learn(1_000_000)
        # model.save(f"./data_n2_v2_set1/0.{i + 1}mil")
        sums = 0
        obs = env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            sums = sums + rewards
        print(sums / 1000)
        sum_list.append(sums / 1000)
    print("average return: ", sum_list)
