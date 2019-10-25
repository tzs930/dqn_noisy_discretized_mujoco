import os
import warnings

import cv2
import numpy as np
from gym import spaces

import argparse
import gym
from pofd import POfD
from stable_baselines.trpo_mpi.trpo_mpi import TRPO
import envs
import stable_baselines.common.tf_util as tf_util

def main():
    """
    Runs the test
    """
    parser = argparse.ArgumentParser(description="Make Expert Demonstrations")
    parser.add_argument('--env', default='HalfCheetahSparse-v0')
    parser.add_argument('--alg', default='trpo')
    parser.add_argument('--load_path', default='demos/demo_HalfCheetahSparse-v0.zip')
    parser.add_argument('--save_path', default='demos/demo_CartPoleSparse-v0.npz')
    args = parser.parse_args()

    env_list = ['HalfCheetahSparse-v0', ]
    alg_list = ['pofd','trpo']
    eval_iteration = 50

    with tf_util.single_threaded_session():
        for envname in env_list:
            env = gym.make(envname)
            for alg in alg_list:
                args.load_path = 'pofd_results_final/%s/%s/0/model.zip' % (alg, envname)
                print(envname, alg, args.load_path)

                if alg == 'pofd':
                    model = POfD.load(args.load_path, env=env)
                else:
                    model = TRPO.load(args.load_path, env=env)
                # import IPython;IPython.embed()
                rews = []
                orews = []
                for i in range(eval_iteration):
                    done = False
                    rewsum = 0
                    original_rews = 0
                    obs = env.reset()
                    while not done:
                        a = model.predict(obs, deterministic=True)[0]
                        obs, rew, done, info = env.step(a)
                        original_rews += info['original_rew']
                        rewsum += rew

                    rews.append(rewsum)
                    orews.append(original_rews)

                # print("- Return Evaluation: %.2f +- .%2f", np.mean(rews))
                with open('evaluation_results.txt', 'a') as file:
                    file.write("- %s, %s Evaluation : %.2f +- %.2f\n" % (envname, alg, np.mean(orews), np.std(orews)))
                # import IPython; IPython.embed()

if __name__ == '__main__':
    main()