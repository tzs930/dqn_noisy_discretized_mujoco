# !/usr/bin/env python3
# noinspection PyUnresolvedReferences
import setGPU
import envs
from mpi4py import MPI
import os
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.trpo_mpi.trpo_mpi import TRPO
from stable_baselines import logger
import stable_baselines.common.tf_util as tf_util
from pofd import POfD
import argparse, gym
from demos.dataset import ExpertDataset

def train(env_id, alg, num_timesteps, seed, demo_data=None, save_path=None, rewcoef=0.1, batchsize=5000):
    """
    Train POfD model for the classic control environment, for testing purposes
    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """
    with tf_util.single_threaded_session():
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            logger.configure()
        else:
            logger.configure(format_strs=[])
            logger.set_level(logger.DISABLED)

        env = gym.make(env_id)
        favor_zero_expert_rewards = False

        if alg == 'pofd':
            model = POfD(MlpPolicy, env, demo_data, timesteps_per_batch=batchsize, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                         entcoeff=0.0, rewcoeff=rewcoef, gamma=0.995, lam=0.97, vf_iters=5, vf_stepsize=1e-3,
                         favor_zero_reward=favor_zero_expert_rewards)
        elif alg == 'trpo':
            model = TRPO(MlpPolicy, env, timesteps_per_batch=batchsize, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                         entcoeff=0.0, gamma=0.995, lam=0.97, vf_iters=5, vf_stepsize=1e-3)
        else:
            raise NotImplementedError

        model.learn(total_timesteps=num_timesteps)

        if save_path is not None:
            model.save(save_path)
        env.close()

def main():
    """
    Runs the test
    """
    parser = argparse.ArgumentParser(description="Train POfD")
    parser.add_argument('--num_timesteps', default=20000001, type=int, help="Maximum number of timesteps")
    parser.add_argument('--alg', default='pofd', help="[pofd|trpo]")
    parser.add_argument('--env', default='CartPoleSparse-v0')
    parser.add_argument('--pid', default=0)
    parser.add_argument('--seed', default=0)
    parser.add_argument('--rewcoef', default=0.1)
    parser.add_argument('--batchsize', default=20000)
    parser.add_argument('--demo_path', default=None)
    parser.add_argument('--num_demo_trajs', default=-1, type=int, help="The number of demonstration trajectories")
    parser.add_argument('--save_path', default=None)
    args = parser.parse_args()

    pid = int(args.pid)
    env_list = ['CartPoleSparse-v0', 'HopperSparse-v0', 'HalfCheetahSparse-v0', 'Walker2dSparse-v0']
    lambda_list = [0.01, 0.01]
    alg_list = ['pofd', 'trpo']

    args.env = env_list[pid // 10]
    args.rewcoef = lambda_list[pid // 10]
    args.alg = alg_list[(pid % 10) // 5]
    args.seed = (pid % 10) % 5

    print(args.env, args.rewcoef, args.alg, args.seed)

    args.save_path = 'results/' + args.alg + '/' + args.env + '/' + str(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    demo_dataset = None
    args.demo_path = 'demos/demo_%s.npz' % args.env

    if args.alg != 'trpo':
        demo_dataset = ExpertDataset(args.demo_path, traj_limitation=args.num_demo_trajs)

    train(args.env, alg=args.alg, num_timesteps=args.num_timesteps, seed=args.seed,
          demo_data=demo_dataset, save_path=args.save_path,
          rewcoef=float(args.rewcoef), batchsize=int(args.batchsize))


if __name__ == '__main__':
    main()