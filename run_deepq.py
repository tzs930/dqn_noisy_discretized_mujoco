# !/usr/bin/env python3
# noinspection PyUnresolvedReferences
import setGPU
import envs
import os
from deepq.dqn import DQN
from stable_baselines.deepq.policies import MlpPolicy
import argparse, gym
import numpy as np

def evaluate(model, env, num_episodes=50):
    rets = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        cur_ret = 0.
        while not done:
            act = model.predict(obs)[0]
            obs, rew, done, _ = env.step(act)
            cur_ret += rew

        rets.append(cur_ret)

    rets = np.array(rets)
    print('* [Epsiode return] Avg : %.2f, Std : %.2f' % (rets.mean(), rets.std()))
            

def train(env_id, num_timesteps, seed, save_path=None, load_path=None, action_noise=0.):
    """
    Train POfD model for the classic control environment, for testing purposes
    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """
    
    env = gym.make(env_id) 
    env.set_action_noise(action_noise)
    model = DQN(MlpPolicy, env)  

    if load_path is not None:
        model.load(load_path)        
    else:
        model.learn(total_timesteps=num_timesteps)
        if save_path is not None:
            model.save(save_path + 'model.zip')

    evaluate(model, env)

    env.close()


def main():
    """
    Runs the test
    """
    parser = argparse.ArgumentParser(description="Train DQN")
    parser.add_argument('--num_timesteps', default=10000001, type=int, help="Maximum number of timesteps")    
    parser.add_argument('--env', default='NoisyDiscretizedReacher-v0')
    parser.add_argument('--pid', default=0)
    parser.add_argument('--seed', default=0)    
    parser.add_argument('--batchsize', default=20000)
    parser.add_argument('--action_noise', default=0.)        
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--load_path', default=None)
    args = parser.parse_args()

    pid = int(args.pid)
    env_list = ['NoisyDiscretizedReacher-v0', 'NoisyDiscretizedSwimmer-v0', 'NoisyDiscretizedHopper-v0']
    # Amount of action noise
    action_noise_list = [0., 0.01, 0.05, 0.1]    

    args.env = env_list[pid // 4]
    args.action_noise = action_noise_list[pid % 4]
    args.seed = 0
    # args.seed = (pid % 12) % 5

    print(args.env, args.action_noise, args.seed)
    args.save_path = 'results/dqn/' + args.env + '/eps' + str(args.action_noise) + '/'
    # args.load_path = 'results/dqn/' + args.env + '/eps' + str(args.action_noise) + '/model.zip'

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)        
    
    train(args.env, num_timesteps=args.num_timesteps, seed=int(args.seed),
          save_path=args.save_path, 
          load_path=args.load_path,
          action_noise=int(args.action_noise))


if __name__ == '__main__':
    main()