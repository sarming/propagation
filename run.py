import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from mpi4py import MPI

import mpi
from simulation import Simulation
import read
import simulation


def build_sim(args):
    if args.graph.endswith('.adjlist'):
        A, node_labels = read.adjlist(args.graph, save_as=args.graph.replace('.adjlist', '.npz'))
    elif args.graph.endswith('.metis'):
        A, node_labels = read.metis(args.graph, zero_based=args.metis_zero_based)
    elif args.graph.endswith('.npz'):
        A, node_labels = read.labelled_graph(args.graph)
    else:
        parser.error(f'Unknown graph file format {args.graph}')

    if args.tweets:
        tweets = read.tweets(args.tweets, node_labels)
        stats = simulation.tweet_statistics(tweets)
        source_map = simulation.tweet_sources(tweets)
    if args.stats:
        stats = read.stats(args.stats)
    if args.source_map:
        source_map = read.source_map(args.source_map)

    if args.params:
        sim = Simulation(A, stats, source_map, params=pd.read_csv(args.params))
    else:
        sim = Simulation(A, stats, source_map)

    if args.discounts:
        sim.params.discount_factor.update(read.single_param(args.discounts))

    return sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runid')
    parser.add_argument('--graph', help="graph file (.npz, .metis or .adjlist)")
    parser.add_argument('--metis_zero_based', action='store_true')
    parser.add_argument('--tweets', help="tweet file (.csv)")
    parser.add_argument('--stats', help="stats file (.csv)")
    parser.add_argument('--params', help="params file (.csv)")
    parser.add_argument('--discounts', help="discounts file (.csv)")
    parser.add_argument('--source_map', help="source map file (.csv)")
    parser.add_argument('--indir', help="input directory (default: data)", default='data')
    parser.add_argument('--outdir', help="output directory (default: data)", default='data')
    parser.add_argument('-f', '--features', help="number of features to simulate", type=int, default=1)
    parser.add_argument('-a', '--sources', help="number of authors per feature class", type=int, default=1)
    parser.add_argument('-s', '--samples', help="number of samples per tweet", type=int, default=1)
    parser.add_argument('--epsilon', help="epsilon for parameter learning", type=float, default=0.001)

    parser.add_argument("command", choices=['learn', 'sim', 'mae'])
    parser.add_argument("topic")
    args = parser.parse_args()

    # Defaults
    if args.runid is None:
        if 'PBS_JOBID' in os.environ:
            args.runid = os.environ.get('PBS_JOBID')
        else:
            args.runid = datetime.now().isoformat()
    if not args.graph:
        args.graph = f'{args.indir}/anonymized_outer_graph_{args.topic}.npz'
    if not args.tweets and not args.stats:
        args.tweets = f'{args.indir}/sim_features_{args.topic}.csv'

    sim = None
    i_am_head = MPI.COMM_WORLD.Get_rank() == 0
    if i_am_head:
        sim = build_sim(args)
        print(f'topic: {args.topic}')
    with mpi.futures(sim, chunksize=1) as sim:
        if sim is not None:
            if args.command == 'learn':
                discount = sim.discount_factor_from_mean_retweets(samples=args.samples, eps=args.epsilon)
                discount.to_csv(f'{args.outdir}/discount-{args.topic}-{args.runid}.csv')
                sim.params.discount_factor.update(discount)
                sim.params.to_csv(f'{args.outdir}/params-{args.topic}-{args.runid}.csv')
            elif args.command == 'mae':
                print(f'{len(sim.features)} features, {args.sources} sources, {args.samples} samples')
                results = pd.DataFrame({'real_retweet_probability': sim.stats.retweet_probability,
                                        'simulation_retweet_probability': np.NaN,
                                        'real_mean_retweets': sim.stats.mean_retweets,
                                        'simulation_mean_retweets': np.NaN,
                                        })
                for feature in sim.features:
                    stats = sim.stats.loc[feature]
                    result = sim.simulate(feature, sources=args.sources, samples=args.samples)
                    # result = mpi.stats_from_futures(result)
                    results.loc[feature].simulation_mean_retweets = result[0]
                    results.loc[feature].simulation_retweet_probability = result[1]
                    print(
                        f'{feature}: {stats.mean_retweets} vs {result[0]}, {stats.retweet_probability} vs {result[1]}')
                results.to_csv(f'{args.outdir}/results-{args.topic}-{args.runid}.csv')
                mae_retweet_probability = results.real_retweet_probability.sub(
                    results.simulation_retweet_probability).abs().mean()
                mae_mean_retweets = results.real_mean_retweets.sub(results.simulation_mean_retweets).abs().mean()
                print(f'MAE: retweet_probability: {mae_retweet_probability} mean_retweets: {mae_mean_retweets}')
                # r = simulate(None, range(100), p=0.0001, corr=0., samples=1000, max_nodes=100)
                # print(r)
            elif args.command == 'sim':
                r = pd.DataFrame(((feature, sim.simulate(feature, sources=args.sources, samples=args.samples))
                                  for feature in sim.sample_feature(args.features)),
                                 columns=['feature', 'results'])
                r[['author_feature', 'tweet_feature']] = pd.DataFrame(r['feature'].tolist())
                r[['mean_retweets', 'retweet_probability']] = pd.DataFrame(r['results'].tolist())
                results = r.groupby(['author_feature', 'tweet_feature']).agg(
                    tweets=('feature', 'size'),
                    mean_retweets=('mean_retweets', 'mean'),
                    retweet_probability=('retweet_probability', 'mean'))
                results.to_csv(f'{args.outdir}/results-{args.topic}-{args.runid}.csv')
                print(results)
