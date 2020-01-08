import abc

import numpy as np
import math
import gym 
from gym.envs.mujoco import ReacherEnv, SwimmerEnv, HopperEnv
from itertools import product

class NoisyEnv:
    def __init__(self, noise_std=0.1):
        self._noise_std=0.1
    
    def set_action_noise(self, noise_std):
        self._noise_std=noise_std
        
class ReacherNoisyDiscretizedEnv(ReacherEnv, NoisyEnv):
    def __init__(self, noise_std=0.1):
        NoisyEnv.__init__(self, noise_std=noise_std)
        ReacherEnv.__init__(self)        

    def _set_action_space(self):        
        n_discretize = 3
        n_actions = 2   
        self.action_space = gym.spaces.Discrete(n_discretize ** n_actions)
        
        return self.action_space        
        
    def step(self, discretized_a):
        n_discretize = 3
        n_actions = 2        
        action_matrix = np.array(list(product(np.linspace(-1., 1., n_discretize), repeat=n_actions)))
        a = action_matrix[discretized_a]

        noisy_action = a + np.random.randn(2) * self._noise_std
        noisy_action = np.clip(noisy_action, [-1., -1.], [1., 1.])
        
        ob, reward, done, info = super().step(noisy_action)
        info['noisy_action'] = noisy_action

        return ob, reward, done, info

class SwimmerNoisyDiscretizedEnv(SwimmerEnv, NoisyEnv):
    def __init__(self, noise_std=0.1):
        NoisyEnv.__init__(self, noise_std=noise_std)
        SwimmerEnv.__init__(self)                        

    def _set_action_space(self):        
        n_discretize = 3
        n_actions = 2   
        self.action_space = gym.spaces.Discrete(n_discretize ** n_actions)

        return self.action_space        
        
    def step(self, discretized_a):
        n_discretize = 3
        n_actions = 2        
        action_matrix = np.array(list(product(np.linspace(-1., 1., n_discretize), repeat=n_actions)))
        a = action_matrix[discretized_a]

        noisy_action = a + np.random.randn(2) * self._noise_std
        noisy_action = np.clip(noisy_action, [-1., -1.], [1., 1.])
                
        ob, reward, done, info = super().step(noisy_action)
        info['noisy_action'] = noisy_action

        return ob,  reward, done, info

class HopperNoisyDiscretizedEnv(HopperEnv, NoisyEnv):
    def __init__(self, noise_std=0.1):
        NoisyEnv.__init__(self, noise_std=noise_std)
        HopperEnv.__init__(self)                        

    def _set_action_space(self):        
        n_discretize = 3
        n_actions = 3   
        self.action_space = gym.spaces.Discrete(n_discretize ** n_actions)
        return self.action_space        
        
    def step(self, discretized_a):
        n_discretize = 3
        n_actions = 3        
        action_matrix = np.array(list(product(np.linspace(-1., 1., n_discretize), repeat=n_actions)))

        a = action_matrix[discretized_a]

        noisy_action = a + np.random.randn(3) * self._noise_std
        noisy_action = np.clip(noisy_action, [-1., -1., -1.], [1., 1., 1.])
                
        ob, reward, done, info = super().step(noisy_action)
        info['noisy_action'] = noisy_action

        return ob,  reward, done, info

