import setGPU
import stable_baselines.trpo_mpi.trpo_mpi as trpo_mpi
import time
from contextlib import contextmanager
from collections import deque
import envs
import gym
from mpi4py import MPI
import tensorflow as tf
import numpy as np

import stable_baselines.common.tf_util as tf_util
from stable_baselines.common import explained_variance, zipsame, dataset, fmt_row, colorize, ActorCriticRLModel, \
    SetVerbosity, TensorboardWriter

from stable_baselines import logger, deepq
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.deepq import DQN
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.trpo_mpi.utils import traj_segment_generator, add_vtarg_and_adv, flatten_lists
from stable_baselines.common.cg import conjugate_gradient

class DQN_early_stopping(DQN):
    def __init__(self, policy, env, base_return, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.1,
                 exploration_final_eps=0.02, train_freq=1, batch_size=32, double_q=True,
                 learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6, param_noise=False, verbose=1, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False):
        super(DQN, self).__init__(policy=policy, env=env, replay_buffer=None,
                                  verbose=verbose, policy_base=DQNPolicy,
                                  requires_vec_env=False, policy_kwargs=policy_kwargs)

        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.double_q = double_q

        self.graph = None
        self.sess = None
        self._train_step = None
        self.step_model = None
        self.update_target = None
        self.act = None
        self.proba_step = None
        self.replay_buffer = None
        self.beta_schedule = None
        self.exploration = None
        self.params = None
        self.summary = None
        self.episode_reward = None

        if _init_setup_model:
            DQN.setup_model(self)

        self.base_return = base_return


    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose):
            self._setup_learn()

            # Create the replay buffer
            if self.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
                if self.prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                else:
                    prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
                self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                    initial_p=self.prioritized_replay_beta0,
                                                    final_p=1.0)
            else:
                self.replay_buffer = ReplayBuffer(self.buffer_size)
                self.beta_schedule = None

            if replay_wrapper is not None:
                assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
                self.replay_buffer = replay_wrapper(self.replay_buffer)

            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                              initial_p=1.0,
                                              final_p=self.exploration_final_eps)

            episode_rewards = [0.0]
            episode_successes = []
            obs = self.env.reset()
            reset = True
            self.episode_reward = np.zeros((1,))

            for _ in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break
                # Take action and update exploration to the newest value
                kwargs = {}
                if not self.param_noise:
                    update_eps = self.exploration.value(self.num_timesteps)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = \
                        -np.log(1. - self.exploration.value(self.num_timesteps) +
                                self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                with self.sess.as_default():
                    action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                env_action = action
                reset = False
                new_obs, rew, done, info = self.env.step(env_action)
                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                episode_rewards[-1] += rew
                if done:
                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    reset = True

                # Do not train if the warmup phase is not over
                # or if there are not enough samples in the replay buffer
                # import IPython;                IPython.embed()
                can_sample = self.replay_buffer.can_sample(self.batch_size)
                if can_sample and self.num_timesteps > self.learning_starts \
                        and self.num_timesteps % self.train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if self.prioritized_replay:
                        experience = self.replay_buffer.sample(self.batch_size,
                                                               beta=self.beta_schedule.value(self.num_timesteps))
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                        weights, batch_idxes = np.ones_like(rewards), None

                    _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                    sess=self.sess)

                    if self.prioritized_replay:
                        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                        self.replay_buffer.update_priorities(batch_idxes, new_priorities)

                if can_sample and self.num_timesteps > self.learning_starts and \
                        self.num_timesteps % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    logger.record_tabular("steps", self.num_timesteps)
                    logger.record_tabular("episodes", num_episodes)
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring",
                                          int(100 * self.exploration.value(self.num_timesteps)))
                    logger.dump_tabular()


                if mean_100ep_reward > self.base_return:
                    break

                self.num_timesteps += 1

        return self

class TRPO_early_stopping(TRPO):
    def __init__(self, policy, env, base_return, gamma=0.99, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, lam=0.98,
                 entcoeff=0.0, cg_damping=1e-2, vf_stepsize=3e-4, vf_iters=3, verbose=1, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=1):
        super(TRPO, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=False,
                               _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                               seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)
        self.using_gail = False
        self.timesteps_per_batch = timesteps_per_batch
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.vf_iters = vf_iters
        self.vf_stepsize = vf_stepsize
        self.entcoeff = entcoeff
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        # GAIL Params
        self.hidden_size_adversary = 100
        self.adversary_entcoeff = 1e-3
        self.expert_dataset = None
        self.g_step = 1
        self.d_step = 1
        self.d_stepsize = 3e-4

        self.graph = None
        self.sess = None
        self.policy_pi = None
        self.loss_names = None
        self.assign_old_eq_new = None
        self.compute_losses = None
        self.compute_lossandgrad = None
        self.compute_fvp = None
        self.compute_vflossandgrad = None
        self.d_adam = None
        self.vfadam = None
        self.get_flat = None
        self.set_from_flat = None
        self.timed = None
        self.allmean = None
        self.nworkers = None
        self.rank = None
        self.reward_giver = None
        self.step = None
        self.proba_step = None
        self.initial_state = None
        self.params = None
        self.summary = None
        self.episode_reward = None

        self.lens_per_iter = []
        self.rews_per_iter = []
        self.truerews_per_iter = []
        self.imitrews_per_iter = []

        if _init_setup_model:
            self.setup_model()

        self.base_return = base_return

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="TRPO",
              reset_num_timesteps=True):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            with self.sess.as_default():
                seg_gen = traj_segment_generator(self.policy_pi, self.env, self.timesteps_per_batch,
                                                 reward_giver=self.reward_giver, gail=self.using_gail)

                episodes_so_far = 0
                timesteps_so_far = 0
                iters_so_far = 0
                t_start = time.time()
                len_buffer = deque(maxlen=40)  # rolling buffer for episode lengths
                reward_buffer = deque(maxlen=40)  # rolling buffer for episode rewards
                self.episode_reward = np.zeros((self.n_envs,))

                true_reward_buffer = None
                if self.using_gail:
                    true_reward_buffer = deque(maxlen=40)

                    # Initialize dataloader
                    batchsize = self.timesteps_per_batch // self.d_step
                    self.expert_dataset.init_dataloader(batchsize)

                    #  Stats not used for now
                    # TODO: replace with normal tb logging
                    #  g_loss_stats = Stats(loss_names)
                    #  d_loss_stats = Stats(reward_giver.loss_name)
                    #  ep_stats = Stats(["True_rewards", "Rewards", "Episode_length"])

                while True:
                    if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) is False:
                            break
                    if total_timesteps and timesteps_so_far >= total_timesteps:
                        break

                    logger.log("********** Iteration %i ************" % iters_so_far)

                    def fisher_vector_product(vec):
                        return self.allmean(self.compute_fvp(vec, *fvpargs, sess=self.sess)) + self.cg_damping * vec

                    # ------------------ Update G ------------------
                    logger.log("Optimizing Policy...")
                    # g_step = 1 when not using GAIL
                    mean_losses = None
                    vpredbefore = None
                    tdlamret = None
                    observation = None
                    action = None
                    seg = None
                    for k in range(self.g_step):
                        with self.timed("sampling"):
                            seg = seg_gen.__next__()
                        add_vtarg_and_adv(seg, self.gamma, self.lam)
                        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                        observation, action = seg["observations"], seg["actions"]
                        atarg, tdlamret = seg["adv"], seg["tdlamret"]


                        vpredbefore = seg["vpred"]  # predicted value function before update
                        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

                        # true_rew is the reward without discount
                        if writer is not None:
                            self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                              seg["true_rewards"].reshape(
                                                                                  (self.n_envs, -1)),
                                                                              seg["dones"].reshape((self.n_envs, -1)),
                                                                              writer, self.num_timesteps)

                        args = seg["observations"], seg["observations"], seg["actions"], atarg
                        # Subsampling: see p40-42 of John Schulman thesis
                        # http://joschu.net/docs/thesis.pdf
                        fvpargs = [arr[::5] for arr in args]

                        self.assign_old_eq_new(sess=self.sess)

                        with self.timed("computegrad"):
                            steps = self.num_timesteps + (k + 1) * (seg["total_timestep"] / self.g_step)
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata() if self.full_tensorboard_log else None
                            # run loss backprop with summary, and save the metadata (memory, compute time, ...)
                            if writer is not None:
                                summary, grad, *lossbefore = self.compute_lossandgrad(*args, tdlamret, sess=self.sess,
                                                                                      options=run_options,
                                                                                      run_metadata=run_metadata)
                                if self.full_tensorboard_log:
                                    writer.add_run_metadata(run_metadata, 'step%d' % steps)
                                writer.add_summary(summary, steps)
                            else:
                                _, grad, *lossbefore = self.compute_lossandgrad(*args, tdlamret, sess=self.sess,
                                                                                options=run_options,
                                                                                run_metadata=run_metadata)

                        lossbefore = self.allmean(np.array(lossbefore))
                        grad = self.allmean(grad)
                        if np.allclose(grad, 0):
                            logger.log("Got zero gradient. not updating")
                        else:
                            with self.timed("conjugate_gradient"):
                                stepdir = conjugate_gradient(fisher_vector_product, grad, cg_iters=self.cg_iters,
                                                             verbose=self.rank == 0 and self.verbose >= 1)
                            assert np.isfinite(stepdir).all()
                            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                            # abs(shs) to avoid taking square root of negative values
                            lagrange_multiplier = np.sqrt(abs(shs) / self.max_kl)
                            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                            fullstep = stepdir / lagrange_multiplier
                            expectedimprove = grad.dot(fullstep)
                            surrbefore = lossbefore[0]
                            stepsize = 1.0
                            thbefore = self.get_flat()
                            thnew = None
                            for _ in range(10):
                                thnew = thbefore + fullstep * stepsize
                                self.set_from_flat(thnew)
                                mean_losses = surr, kl_loss, *_ = self.allmean(
                                    np.array(self.compute_losses(*args, sess=self.sess)))
                                improve = surr - surrbefore
                                logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                                if not np.isfinite(mean_losses).all():
                                    logger.log("Got non-finite value of losses -- bad!")
                                elif kl_loss > self.max_kl * 1.5:
                                    logger.log("violated KL constraint. shrinking step.")
                                elif improve < 0:
                                    logger.log("surrogate didn't improve. shrinking step.")
                                else:
                                    logger.log("Stepsize OK!")
                                    break
                                stepsize *= .5
                            else:
                                logger.log("couldn't compute a good step")
                                self.set_from_flat(thbefore)
                            if self.nworkers > 1 and iters_so_far % 20 == 0:
                                # list of tuples
                                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), self.vfadam.getflat().sum()))
                                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

                        with self.timed("vf"):
                            for _ in range(self.vf_iters):
                                # NOTE: for recurrent policies, use shuffle=False?
                                for (mbob, mbret) in dataset.iterbatches((seg["observations"], seg["tdlamret"]),
                                                                         include_final_partial_batch=False,
                                                                         batch_size=128,
                                                                         shuffle=True):
                                    grad = self.allmean(self.compute_vflossandgrad(mbob, mbob, mbret, sess=self.sess))
                                    self.vfadam.update(grad, self.vf_stepsize)

                    for (loss_name, loss_val) in zip(self.loss_names, mean_losses):
                        logger.record_tabular(loss_name, loss_val)

                    logger.record_tabular("explained_variance_tdlam_before",
                                          explained_variance(vpredbefore, tdlamret))

                    if self.using_gail:
                        # ------------------ Update D ------------------
                        logger.log("Optimizing Discriminator...")
                        logger.log(fmt_row(13, self.reward_giver.loss_name))
                        assert len(observation) == self.timesteps_per_batch
                        batch_size = self.timesteps_per_batch // self.d_step

                        # NOTE: uses only the last g step for observation
                        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
                        # NOTE: for recurrent policies, use shuffle=False?
                        for ob_batch, ac_batch in dataset.iterbatches((observation, action),
                                                                      include_final_partial_batch=False,
                                                                      batch_size=batch_size,
                                                                      shuffle=True):
                            ob_expert, ac_expert = self.expert_dataset.get_next_batch()
                            # update running mean/std for reward_giver
                            if self.reward_giver.normalize:
                                self.reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))

                            # Reshape actions if needed when using discrete actions
                            if isinstance(self.action_space, gym.spaces.Discrete):
                                if len(ac_batch.shape) == 2:
                                    ac_batch = ac_batch[:, 0]
                                if len(ac_expert.shape) == 2:
                                    ac_expert = ac_expert[:, 0]
                            *newlosses, grad = self.reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
                            self.d_adam.update(self.allmean(grad), self.d_stepsize)
                            d_losses.append(newlosses)
                        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

                        # lr: lengths and rewards
                        lr_local = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
                        list_lr_pairs = MPI.COMM_WORLD.allgather(lr_local)  # list of tuples
                        lens, rews, true_rets = map(flatten_lists, zip(*list_lr_pairs))
                        true_reward_buffer.extend(true_rets)
                    else:
                        # lr: lengths and rewards
                        lr_local = (seg["ep_lens"], seg["ep_rets"])  # local values
                        list_lr_pairs = MPI.COMM_WORLD.allgather(lr_local)  # list of tuples
                        lens, rews = map(flatten_lists, zip(*list_lr_pairs))
                    len_buffer.extend(lens)
                    reward_buffer.extend(rews)

                    if len(len_buffer) > 0:
                        logger.record_tabular("EpLenMean", np.mean(len_buffer))
                        logger.record_tabular("EpRewMean", np.mean(reward_buffer))
                    if self.using_gail:
                        logger.record_tabular("EpTrueRewMean", np.mean(true_reward_buffer))
                    logger.record_tabular("EpThisIter", len(lens))
                    episodes_so_far += len(lens)
                    current_it_timesteps = MPI.COMM_WORLD.allreduce(seg["total_timestep"])
                    timesteps_so_far += current_it_timesteps
                    self.num_timesteps += current_it_timesteps
                    iters_so_far += 1

                    logger.record_tabular("EpisodesSoFar", episodes_so_far)
                    logger.record_tabular("TimestepsSoFar", self.num_timesteps)
                    logger.record_tabular("TimeElapsed", time.time() - t_start)

                    if np.mean(reward_buffer) >= self.base_return:
                        break

                    if self.verbose >= 1 and self.rank == 0:
                        logger.dump_tabular()

        return self

def train(env_id, num_timesteps, seed, base_return, save_path):
    # env = make_mujoco_env(env_id, workerseed)
    env = gym.make(env_id)
    # model = trpo_mpi.TRPO(MlpPolicy, env, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10,
    #                             cg_damping=0.1, entcoeff=0.0,
    #                             gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    model = TRPO_early_stopping(MlpPolicy, env, base_return, timesteps_per_batch=5000,
                                max_kl=0.01, cg_iters=10, cg_damping=0.1, entcoeff=0.0,
                                gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    # model = DQN_early_stopping(env=env,
    #                            policy=MlpPolicy,
    #                            base_return=base_return,
    #                            learning_rate=1e-3,
    #                            buffer_size=50000,
    #                            exploration_fraction=0.1,
    #                            exploration_final_eps=0.02)
    # import IPython; IPython.embed()
    model.learn(total_timesteps=num_timesteps)
    env.close()
    model.save(save_path)

    return model

import argparse
import os

def main():
    """
    Runs the test
    """
    parser = argparse.ArgumentParser(description="Train a demonstration policy with TRPO algorithm")
    parser.add_argument('--num_timesteps', default=100000000, type=int, help="Maximum number of timesteps")
    parser.add_argument('--env', default='CartPole-v1')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--base_return', default=-100)
    parser.add_argument('--save_path', default='cartpole_demo')
    args = parser.parse_args()
    args.base_return = float(args.base_return)
    # Training

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    model = train(args.env, num_timesteps=args.num_timesteps,
                  seed=args.seed, base_return=float(args.base_return),
                  save_path=args.save_path)

    # Testing
    env = gym.make(args.env)
    rews = []
    for i in range(50):
        done = False
        rewsum = 0
        obs = env.reset()
        while not done:
            a = model.predict(obs)[0]
            obs, rew, done, _ = env.step(a)
            rewsum += rew
        rews.append(rewsum)
    print("- Return Evaluation: ", np.mean(rews))
    # import IPython; IPython.embed()

if __name__ == '__main__':
    main()