import os
import sys

import pandas as pd
from mpi4py import MPI
import numpy as np
import scipy as sp
from mpi import mpi_futures

from simulation import Simulation


def read_discount(file):
    discount = pd.read_csv(file, dtype={'author_feature': str, 'tweet_feature': str})
    discount.set_index(['author_feature', 'tweet_feature'], inplace=True)
    return discount.squeeze()


if __name__ == "__main__":
    sim = None
    i_am_head = MPI.COMM_WORLD.Get_rank() == 0
    if i_am_head:
        datadir = 'data'
        topic = 'bvb_20200409'
        if len(sys.argv) > 1:
            topic = sys.argv[1]
        print(f'topic: {topic}')
        # A, node_labels = read.labelled_graph(f'{datadir}/outer_neos.npz')
        sim = Simulation.from_files(f'{datadir}/outer_{topic}.npz', f'{datadir}/sim_features_{topic}.csv')

        sim.params.discount_factor.update(read_discount(f'{datadir}/discount-{topic}.csv'))

    with mpi_futures(sim, num_chunks=16000) as sim:
        if sim is not None:
            # discount = sim.discount_factor_from_mean_retweets(samples=1, eps=1)
            # discount = sim.discount_factor_from_mean_retweets(samples=8000, eps=0.001)
            # sim.params.discount_factor.update(discount)
            # print(discount.to_csv())
            # discount.to_csv(f'{datadir}/discount-{topic}-{os.environ.get("PBS_JOBID")}.csv')
            print(sim.params.to_csv())
            sim.params.to_csv(f'{datadir}/params-{topic}-{os.environ.get("PBS_JOBID")}.csv')
            # sys.exit()
            sources = 1000
            samples = 8000
            print(f'{len(sim.features)} features, {sources} sources, {samples} samples')
            results = pd.DataFrame({'real_retweet_probability': sim.stats.retweet_probability,
                                    'simulation_retweet_probability': np.NaN,
                                    'real_mean_retweets': sim.stats.mean_retweets,
                                    'simulation_mean_retweets': np.NaN,
                                    })
            for feature in sim.features:
                stats = sim.stats.loc[feature]
                result = sim.simulate(feature, sources=sources, samples=samples)
                results.loc[feature].simulation_mean_retweets = result[0]
                results.loc[feature].simulation_retweet_probability = result[1]
                print(f'{feature}: {stats.mean_retweets} vs {result[0]}, {stats.retweet_probability} vs {result[1]}')
            results.to_csv(f'{datadir}/results-{topic}-{os.environ.get("PBS_JOBID")}.csv')
            mae_retweet_probability = results.real_retweet_probability.sub(
                results.simulation_retweet_probability).abs().mean()
            mae_mean_retweets = results.real_mean_retweets.sub(results.simulation_mean_retweets).abs().mean()
            print(f'MAE: retweet_probability: {mae_retweet_probability} mean_retweets: {mae_mean_retweets}')
            # r = simulate(None, range(100), p=0.0001, corr=0., samples=1000, max_nodes=100)
            # print(r)
