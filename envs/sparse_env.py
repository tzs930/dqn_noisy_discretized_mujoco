import abc

import numpy as np
from gym.envs.mujoco import HopperEnv, HalfCheetahEnv, Walker2dEnv, InvertedDoublePendulumEnv, HumanoidEnv, ReacherEnv
from gym.envs.mujoco.humanoid import mass_center
from stable_baselines import logger


class InvertedDoublePendulumSparseEnv(InvertedDoublePendulumEnv):
    def __init__(self):
        InvertedDoublePendulumEnv.__init__(self)

    def step(self, a):
        a = self._add_noise(a)
        ob, reward, done, info = super().step(a)

        sparse_rew = 1 if ob[3] > 0.89 else 0
        info['original_rew'] = reward
        return ob, sparse_rew, done, info


class HopperSparseEnv(HopperEnv):
    def __init__(self):
        self._unit = 1.
        self._base_pos = self._unit
        HopperEnv.__init__(self)

    def step(self, a):
        ob, reward, done, info = super().step(a)
        pos_after = self.sim.data.qpos[0]

        if pos_after > self._base_pos:
            sparse_rew = int((pos_after - self._base_pos) / self._unit) + 1
            self._base_pos += self._unit * sparse_rew
            sparse_rew = 1
        else:
            sparse_rew = 0

        info['original_rew'] = reward

        # if done:
        #     return ob, -10., done, info

        return self._handle_step_results_for_unbiasing(ob, sparse_rew, done, info)

    def reset(self):
        self._base_pos = self._unit
        return super().reset()


class Walker2dSparseEnv(Walker2dEnv, NoisySparseEnv):
    def __init__(self):
        self._base_pos = 1.
        Walker2dEnv.__init__(self)

    def step(self, a):
        ob, reward, done, info = super().step(a)
        pos_after = self.sim.data.qpos[0]

        if pos_after > self._base_pos:
            self._base_pos += 1.
            sparse_rew = 1
        else:
            sparse_rew = 0

        info['original_rew'] = reward
        return ob, sparse_rew, done, info

    def reset(self):
        self._base_pos = 1.
        return super().reset()


class HalfCheetahSparseEnv(HalfCheetahEnv, NoisySparseEnv):
    def __init__(self):
        self._base_pos = 15.
        HalfCheetahEnv.__init__(self)

    def step(self, a, ac_noise=0.):
        # a = self._add_noise(a)
        ob, reward, done, info = super().step(a)
        pos_after = self.sim.data.qpos[0]

        if pos_after > self._base_pos:
            self._base_pos += 15.
            sparse_rew = 1
        else:
            sparse_rew = 0

        info['original_rew'] = reward
        return ob, sparse_rew, done, info

    def reset(self):
        self._base_pos = 15.
        return super().reset()


class HumanoidSparseEnv(HumanoidEnv, NoisySparseEnv):
    def __init__(self):
        self._base_pos = 1.
        HumanoidEnv.__init__(self)

    def step(self, a):
        # a = self._add_noise(a)
        ob, reward, done, info = super().step(a)
        mass_center_after = mass_center(self.model, self.sim)

        if mass_center_after > self._base_pos:
            self._base_pos += 1.
            sparse_rew = 1
        else:
            sparse_rew = 0

        info['original_rew'] = reward
        return ob, sparse_rew, done, info

    def reset(self):
        self._base_pos = 1.
        return super().reset()


class ReacherSparseEnv(ReacherEnv):
    def __init__(self):
        ReacherEnv.__init__(self)
        self._hit_timesteps = 0

    def step(self, a):
        ob, reward, done, info = super().step(a)
        distance = np.linalg.norm(self.get_body_com('fingertip') - self.get_body_com('target'))

        sparse_rew = 0.
        if distance < 1e-2:
            self._hit_timesteps += 1
            if self._hit_timesteps >= 5:
                sparse_rew = 1.
                done = True
        else:
            self._hit_timesteps = 0

        info['original_rew'] = reward
        info['dense_dist'] = -distance
        return ob, sparse_rew, done, info

    def reset(self):
        self._hit_timesteps = 0
        return super().reset()

