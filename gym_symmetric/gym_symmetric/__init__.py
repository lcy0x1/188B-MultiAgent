from gym.envs.registration import register

register(
    id='symmetric-v0',
    entry_point='gym_symmetric.envs:VehicleEnv',
)