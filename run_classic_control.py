# !/usr/bin/env python3
# noinspection PyUnresolvedReferences
import setGPU
from mpi4py import MPI
import os
from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.gail import GAIL
from stable_baselines.trpo_mpi.trpo_mpi import TRPO
from stable_baselines import logger
import stable_baselines.common.tf_util as tf_util
from pofd import POfD
import argparse, gym
from demos.dataset import ExpertDataset

def train(env_id, alg, num_timesteps, seed, demo_data=None, save_path=None):
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
        # assert demo_data is None, "To train with POfD, demonstration trajectories are needed!"
        # import IPython;IPython.embed()
        batchsize = 5000

        if alg == 'pofd':
            model = POfD(MlpPolicy, env, demo_data, timesteps_per_batch=batchsize, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                         entcoeff=0.0, rewcoeff=0.01, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
        elif alg == 'gail':
            model = GAIL(MlpPolicy, env, demo_data, timesteps_per_batch=batchsize, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                         entcoeff=0.0, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
        elif alg == 'trpo':
            model = TRPO(MlpPolicy, env, timesteps_per_batch=batchsize, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                         entcoeff=0.0, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
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
    parser.add_argument('--num_timesteps', default=500000, type=int, help="Maximum number of timesteps")
    parser.add_argument('--alg', default='pofd', help="[pofd|gail|trpo]")
    parser.add_argument('--env', default='CartPole-v1')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--demo_path', default='demos/demo_cartpole.npz')
    parser.add_argument('--save_path', default=None)
    args = parser.parse_args()
    args.save_path = 'results/' + args.alg + '/' + args.env + '/' + str(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    demo_dataset = None

    if args.alg != 'trpo':
        demo_dataset = ExpertDataset(args.demo_path)

    train(args.env, alg=args.alg, num_timesteps=args.num_timesteps, seed=args.seed,
          demo_data=demo_dataset, save_path=args.save_path)





if __name__ == '__main__':
    main()