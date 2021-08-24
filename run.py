#!/usr/bin/env python
import argparse
import os
import subprocess
import time
from datetime import datetime

import pandas as pd
from mpi4py import MPI

import mpi
import optimize
import propagation
import read
import simulation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--runid')
    parser.add_argument('--graph', help="graph file (.npz, .metis or .adjlist)")
    parser.add_argument('--metis_zero_based', action='store_true')
    parser.add_argument('--tweets', help="tweet file (.csv)")
    parser.add_argument('--stats', help="stats file (.csv)")
    parser.add_argument('--params', help="params file (.csv)")
    parser.add_argument('--discounts', help="discounts file (.csv)")
    parser.add_argument('--corrs', help="corr file (.csv)")
    parser.add_argument('--source_map', help="source map file (.csv)")
    parser.add_argument('--indir', help="input directory (default: data)", default='data')
    parser.add_argument('--outdir', help="output directory (default: out)", default='out')
    parser.add_argument('-f', '--features', help="number of features to simulate", type=int, default=1)
    parser.add_argument('-a', '--sources', help="number of authors per feature class", type=int, default=1)
    parser.add_argument('-s', '--samples', help="number of samples per tweet", type=int, default=1)
    parser.add_argument('--epsilon', help="epsilon for parameter learning", type=float, default=0.001)
    parser.add_argument('--max_depth', help="maximum depth to simulate", type=int)
    parser.add_argument('--max_nodes', help="maximum retweet count to simulate", type=int)
    parser.add_argument('--sample_split', help="split tasks (default: 1, has to divide 'samples', changes RNG)",
                        type=int, default=1)
    parser.add_argument('--seed', help="seed for RNG", type=int)
    parser.add_argument("command", choices=['learn_discount', 'learn_corr', 'optimize', 'sim', 'simtweets', 'mae'])
    parser.add_argument("topic")
    args = parser.parse_args()

    # Defaults
    if args.runid is None:
        if 'PBS_JOBID' in os.environ:
            args.runid = os.environ.get('PBS_JOBID')
        else:
            args.runid = datetime.now().isoformat()
    if not args.graph:
        args.graph = f'{args.indir}/anon_graph_inner_{args.topic}.npz'
        if not os.path.isfile(args.graph):
            args.graph = args.graph.replace('npz', 'metis')
    # if not args.tweets and not args.stats: TODO
    if not args.tweets:
        args.tweets = f'{args.indir}/sim_features_{args.topic}.csv'
    if not args.tweets and not args.source_map:
        raise ValueError("Either --tweets or --source_map has to be provided!")
    if args.samples % args.sample_split != 0:
        raise ValueError("sample_split has to evenly divide samples!")

    return args


def build_sim(args):
    """ Construct simulation object. """
    if args.graph.endswith('.adjlist'):
        A, node_labels = read.adjlist(args.graph, save_as=args.graph.replace('.adjlist', '.adjlist.npz'))
    elif args.graph.endswith('.metis'):
        A, node_labels = read.metis(args.graph, zero_based=args.metis_zero_based,
                                    save_as=args.graph.replace('.metis', '.npz'))
    elif args.graph.endswith('.npz'):
        A, node_labels = read.labelled_graph(args.graph)
        if args.tweets:
            print('Assuming METIS style ids.')
    else:
        raise ValueError(f"Unknown graph file format {args.graph}.\nPress Ctrl-C to terminate MPI procs.")

    # Input files
    if args.tweets:
        id_type = 'adjlist' if args.graph.endswith('.adjlist') else 'metis'
        tweets = read.tweets(args.tweets, node_labels, id_type)
        stats = simulation.tweet_statistics(tweets)
        source_map = simulation.tweet_sources(tweets)
    if args.stats:
        stats = read.stats(args.stats)
    if args.source_map:
        source_map = read.source_map(args.source_map)

    if args.params:
        sim = simulation.Simulation(A, stats, source_map, params=pd.read_csv(args.params), seed=args.seed)
    else:
        sim = simulation.Simulation(A, stats, source_map, seed=args.seed)

    if args.max_depth:
        sim.params.max_depth = args.max_depth
    if args.max_nodes:
        sim.params.max_nodes = args.max_nodes

    if args.discounts:
        sim.params['discount_factor'] = read.single_param(args.discounts)

    if args.corrs:
        sim.params['corr'] = read.single_param(args.corrs)

    return sim


def agg_statistics(feature_results):
    """Aggregate statistics by feature.

    Args:
        feature_results: iterable of (feature, (mean_retweets, retweet_probability))

    Returns:
        DataFrame with aggregated number of tweets, mean_retweets and retweet_probability.

    """
    r = pd.DataFrame(feature_results, columns=['feature', 'results'])
    r[['author_feature', 'tweet_feature']] = pd.DataFrame(r['feature'].tolist())
    r[['mean_retweets', 'retweet_probability']] = pd.DataFrame(r['results'].tolist())
    stats = r.groupby(['author_feature', 'tweet_feature']).agg(
        tweets=('feature', 'size'),  # TODO: multiply with authors * samples
        mean_retweets=('mean_retweets', 'mean'),
        retweet_probability=('retweet_probability', 'mean'))
    return stats


def explode_tweets(tweet_results):
    r = pd.DataFrame(tweet_results, columns=['feature', 'results'])
    r[['author_feature', 'tweet_feature']] = pd.DataFrame(r['feature'].tolist())

    r['results'] = r['results'].apply(list)
    r = r.explode('results', ignore_index=True)

    r[['author', 'retweets']] = pd.DataFrame(r['results'].tolist())

    r['retweets'] = r['retweets'].apply(list)
    r = r.explode('retweets', ignore_index=True)

    return r[['author', 'author_feature', 'tweet_feature', 'retweets']]


def main():
    args = parse_args()  # Put this here to terminate all MPI procs on parse errors

    sim = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"mpi_size: {MPI.COMM_WORLD.Get_size()}")
        print(f"mpi_vendor: {MPI.get_vendor()}")
        code_version = subprocess.run(['git', 'describe', '--tags', '--dirty'], capture_output=True, text=True).stdout
        print(f'code_version: {code_version.rstrip()}')

        t = time.time()
        sim = build_sim(args)
        print(f"readtime: {time.time() - t}")
        t = time.time()

        sim.stats.sort_values(by=['mean_retweets', 'retweet_probability'], ascending=False, inplace=True)
        sim.features = sim.stats.index
        sim.params = sim.params.reindex(index=sim.features)

        print(f"args: {args}")
        print(f"seed: {sim.seed.entropy}")
        print(f"topic: {args.topic}")
    else:
        propagation.compile()

    # if True: # bypass mpi
    with mpi.futures(sim, chunksize=1, sample_split=args.sample_split) as sim:
        if sim is not None:
            print(f"setuptime: {time.time() - t}")
            t = time.time()

            if args.command == 'learn_discount':
                discount = sim.discount_factor_from_mean_retweets(samples=args.samples, eps=args.epsilon)
                discount.to_csv(f'{args.outdir}/discount-{args.topic}-{args.runid}.csv')
                sim.params['discount_factor'] = discount
                sim.params.to_csv(f'{args.outdir}/params-{args.topic}-{args.runid}.csv')

            elif args.command == 'learn_corr':
                corr = sim.corr_from_mean_retweets(samples=args.samples, eps=args.epsilon)
                corr.to_csv(f'{args.outdir}/corr-{args.topic}-{args.runid}.csv')
                sim.params['corr'] = corr
                sim.params.to_csv(f'{args.outdir}/params-{args.topic}-{args.runid}.csv')

            elif args.command == 'optimize':
                optimize.optimize(sim, sources=None if args.sources < 1 else args.sources, samples=args.samples)
                sim.params.to_csv(f'{args.outdir}/params-{args.topic}-{args.runid}.csv')

            elif args.command == 'mae':
                print(f'{len(sim.features)} features, {args.sources} sources, {args.samples} samples')
                r = agg_statistics((feature, sim.simulate(feature, sources=args.sources, samples=args.samples))
                                   for feature in sim.features)

                # assert r.index.equals(sim.stats.index)
                # r = r.reindex(index=sim.features)
                r['real_mean_retweets'] = sim.stats.mean_retweets
                r['real_retweet_probability'] = sim.stats.retweet_probability
                # r.rename(columns={'mean_retweets': 'simulation_mean_retweets',
                #                   'retweet_probability': 'simulation_retweet_probability'}, inplace=True)

                for feature, i in r.iterrows():
                    print(f'{feature}: '
                          f'{i.real_mean_retweets} vs {i.mean_retweets}, '
                          f'{i.real_retweet_probability} vs {i.retweet_probability}')
                r.to_csv(f'{args.outdir}/results-{args.topic}-{args.runid}.csv')
                sim.params.to_csv(f'{args.outdir}/params-{args.topic}-{args.runid}.csv')

                mae_retweet_probability = r.real_retweet_probability.sub(r.retweet_probability).abs().mean()
                mae_mean_retweets = r.real_mean_retweets.sub(r.mean_retweets).abs().mean()
                print(f"MAE: retweet_probability: {mae_retweet_probability} mean_retweets: {mae_mean_retweets}")

            elif args.command == 'sim':
                r = agg_statistics((feature, sim.simulate(feature, sources=args.sources, samples=args.samples))
                                   for feature in sim.sample_feature(args.features))
                r.to_csv(f'{args.outdir}/results-{args.topic}-{args.runid}.csv')
                print(r)

            elif args.command == 'simtweets':
                r = explode_tweets(sim.run(num_features=args.features, num_sources=args.sources, samples=args.samples))
                r.to_csv(f'{args.outdir}/results-{args.topic}-{args.runid}.csv')
                print(r)
            print(f"runtime: {time.time() - t}")


if __name__ == "__main__":
    # sim.simulate(,{edge_probability:0.1})
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)

    if MPI.COMM_WORLD.Get_rank() == 0:
        startTime = time.time()
    main()
    if MPI.COMM_WORLD.Get_rank() == 0:
        endTime = time.time()
        print("Total Time Elapsed: " + str(endTime - startTime))
