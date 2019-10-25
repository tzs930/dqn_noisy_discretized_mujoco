from gym.envs.registration import register

register(
    id='CartPoleSparse-v0',
    entry_point='envs.sparse_env:CartPoleSparseEnv',
    max_episode_steps=500,
)

register(
    id='MountainCarSparse-v0',
    entry_point='envs.sparse_env:MountainCarSparseEnv',
    max_episode_steps=200,
)

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
    id='InvertedDoublePendulumSparse-v0',
    entry_point='envs.sparse_env:InvertedDoublePendulumSparseEnv',
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