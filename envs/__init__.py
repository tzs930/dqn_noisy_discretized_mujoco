from gym.envs.registration import register
​
# Noise std = 0.1
register(
    id='Walker2dSparse-v0',
    entry_point='envs.sparse_env:Walker2dSparseEnv',
    max_episode_steps=1000,
)

register(
    id='HopperSparse-v0',
    entry_point='envs.sparse_env:HopperSparseEnv',
    max_episode_steps=1000,
)

register(
    id='HalfCheetahSparse-v0',
    entry_point='envs.sparse_env:HalfCheetahSparseEnv',
    max_episode_steps=1000,
)

register(
    id='HumanoidSparse-v0',
    entry_point='envs.sparse_env:HumanoidSparseEnv',
    max_episode_steps=1000,
)

register(
    id='ReacherSparse-v0',
    entry_point='envs.sparse_env:ReacherSparseEnv',
    max_episode_steps=500,
)