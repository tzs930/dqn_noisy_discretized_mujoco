import abc

import numpy as np
import math
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.classic_control.mountain_car import MountainCarEnv
from gym.envs.mujoco import HopperEnv, HalfCheetahEnv, Walker2dEnv, InvertedDoublePendulumEnv, HumanoidEnv, ReacherEnv
from gym.envs.mujoco.humanoid import mass_center
from stable_baselines import logger

class CartPoleSparseEnv(CartPoleEnv):
    def __init__(self):
        CartPoleEnv.__init__(self)
        self.steps_beyond_done = 0
        self.success_steps = 0

    def reset(self):
        obs = CartPoleEnv.reset(self)
        self.steps_beyond_done = 0
        self.success_steps = 0
        return obs

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x, x_dot, theta, theta_dot)
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            self.success_steps += 1
            original_rew = 1
        else:
            self.success_steps = 0
            if self.steps_beyond_done > 0:
                original_rew = 1
            else:
                original_rew = 0

            self.steps_beyond_done += 1

        if self.success_steps >= 200:
            reward = 1
        else:
            reward = 0

        info = {}
        info['original_rew'] = original_rew

        return np.array(self.state), reward, done, info


class MountainCarSparseEnv(MountainCarEnv):
    def __init__(self):
        MountainCarEnv.__init__(self)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        if done:
            reward = 1.0
        else:
            reward = 0.0

        info = {}
        info['original_rew'] = -1.

        self.state = (position, velocity)
        return np.array(self.state), reward, done, info

class InvertedDoublePendulumSparseEnv(InvertedDoublePendulumEnv):
    def __init__(self):
        InvertedDoublePendulumEnv.__init__(self)

    def step(self, a):
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

        return ob, sparse_rew, done, info

    def reset(self):
        self._base_pos = self._unit
        return super().reset()


class Walker2dSparseEnv(Walker2dEnv):
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


class HalfCheetahSparseEnv(HalfCheetahEnv):
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


class HumanoidSparseEnv(HumanoidEnv):
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

