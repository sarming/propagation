#!/usr/bin/env python
import argparse
import os
from datetime import datetime

import pandas as pd
from mpi4py import MPI

import mpi
import read
import simulation
from simulation import Simulation


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
    parser.add_argument('--outdir', help="output directory (default: data)", default='data')
    parser.add_argument('-f', '--features', help="number of features to simulate", type=int, default=1)
    parser.add_argument('-a', '--sources', help="number of authors per feature class", type=int, default=1)
    parser.add_argument('-s', '--samples', help="number of samples per tweet", type=int, default=1)
    parser.add_argument('--epsilon', help="epsilon for parameter learning", type=float, default=0.001)
    parser.add_argument('--max_depth', help="maximum depth to simulate", type=int)
    parser.add_argument('--max_nodes', help="maximum retweet count to simulate", type=int)
    parser.add_argument("command", choices=['learn_discount','learn_corr', 'sim', 'mae'])
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
    #if not args.tweets and not args.stats:
    if not args.tweets:
        args.tweets = f'{args.indir}/sim_features_{args.topic}.csv'

    return args


def build_sim(args):
    """ Construct simulation object. """
    if args.graph.endswith('.adjlist'):
        A, node_labels = read.adjlist(args.graph, save_as=args.graph.replace('.adjlist', '.npz'))
    elif args.graph.endswith('.metis'):
        A, node_labels = read.metis(args.graph, zero_based=args.metis_zero_based)
    elif args.graph.endswith('.npz'):
        A, node_labels = read.labelled_graph(args.graph)
    else:
        raise ValueError(f'Unknown graph file format {args.graph}.\nPress Ctrl-C to terminate MPI procs.')

    # Input files
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


def main():
    args = parse_args()  # Put this here to terminate all MPI procs on parse errors

    sim = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        sim = build_sim(args)
        print(f'args: {args}')
        print(f'topic: {args.topic}')

    # if True: # bypass mpi
    with mpi.futures(sim, chunksize=1) as sim:
        if sim is not None:
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

            elif args.command == 'mae':
                print(f'{len(sim.features)} features, {args.sources} sources, {args.samples} samples')
                r = agg_statistics((feature, sim.simulate(feature, sources=args.sources, samples=args.samples))
                                   for feature in sim.features)

                assert r.index.equals(sim.stats.index)
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
                print(f'MAE: retweet_probability: {mae_retweet_probability} mean_retweets: {mae_mean_retweets}')

            elif args.command == 'sim':
                r = agg_statistics((feature, sim.simulate(feature, sources=args.sources, samples=args.samples))
                                   for feature in sim.sample_feature(args.features))
                r.to_csv(f'{args.outdir}/results-{args.topic}-{args.runid}.csv')
                print(r)


if __name__ == "__main__":
    main()
