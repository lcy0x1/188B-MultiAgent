from gym.envs.registration import register

register(
    id='symmetric-v1',
    entry_point='gym_symmetric.envs:VehicleEnv',
)