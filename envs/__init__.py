from gym.envs.registration import register

register(
    id='NoisyDiscretizedSwimmer-v0',
    entry_point='envs.stochastic_mujoco:SwimmerNoisyDiscretizedEnv',
    max_episode_steps=1000,
    reward_threshold=360.0
)

register(
    id='NoisyDiscretizedHopper-v0',
    entry_point='envs.stochastic_mujoco:HopperNoisyDiscretizedEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0
)

register(
    id='NoisyDiscretizedReacher-v0',
    entry_point='envs.stochastic_mujoco:ReacherNoisyDiscretizedEnv',
    max_episode_steps=50,
    reward_threshold=-3.75
)