import matplotlib.pyplot as plt
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot POfD")
    parser.add_argument('--env', default='HopperSparse-v0')
    parser.add_argument('--maxseed', default=5)
    parser.add_argument('--maxiter', default=1000)
    args = parser.parse_args()

    plt.figure(figsize=(14, 3))

    max_iter = [300, 500, 1000, 500]
    behavior_return = [51, 772.93, 1732.92, 2175.71]
    env_list = ['CartPoleSparse-v0', 'HopperSparse-v0', 'HalfCheetahSparse-v0', 'Walker2dSparse-v0']
    ax_list = [141, 142, 143, 144]

    for i, env in enumerate(env_list):
        args.env = env
        args.maxiter = max_iter[i]

        # TRPO
        trpo_paths = ['pofd_results/trpo/%s/%d/eprews.npz' % (args.env, i) for i in range(args.maxseed)]
        trpo_denserews = []
        for path in trpo_paths:
            data = np.load(path)
            trpo_denserews.append(data['denserews'])

        trpo_denserews = np.array(trpo_denserews)[:, :args.maxiter]
        trpo_denserews_mean = trpo_denserews.mean(axis=0)
        trpo_denserews_stderr = trpo_denserews.std(axis=0) / np.sqrt(args.maxseed)

        # POFD
        pofd_paths = ['pofd_results/pofd/%s/%d/eprews.npz' % (args.env, i) for i in range(args.maxseed)]
        pofd_truerews = []
        pofd_rews = []
        pofd_imitrews = []
        pofd_denserews = []

        for path in pofd_paths:
            data = np.load(path)
            pofd_truerews.append(data['truerews'])
            pofd_rews.append(data['rews'])
            pofd_imitrews.append(data['imitrews'])
            pofd_denserews.append(data['denserews'])

        pofd_denserews = np.array(pofd_denserews)[:, :args.maxiter]
        pofd_denserews_mean = pofd_denserews.mean(axis=0)
        pofd_denserews_stderr = pofd_denserews.std(axis=0) / np.sqrt(args.maxseed)

        ax = plt.subplot(ax_list[i])
        plt.plot(np.arange(args.maxiter), trpo_denserews_mean, color=[0.7, 0., 0., 1.])
        plt.fill_between(np.arange(args.maxiter), trpo_denserews_mean - trpo_denserews_stderr,
                         trpo_denserews_mean + trpo_denserews_stderr, color=[0.7, 0., 0., .5])

        plt.plot(np.arange(args.maxiter), pofd_denserews_mean, color=[0., 0., 0.7, 1.])
        plt.fill_between(np.arange(args.maxiter), pofd_denserews_mean - pofd_denserews_stderr,
                         pofd_denserews_mean + pofd_denserews_stderr, color=[0., 0., 0.7, .5])

        plt.axhline(y=behavior_return[i], color='y')
        plt.title(args.env)

    plt.savefig('figure1.png')
    plt.show()


if __name__ == '__main__':
    main()