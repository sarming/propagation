#!/usr/bin/env python
import argparse
import os
import pickle
import resource
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
from mpi4py import MPI

# https://stackoverflow.com/a/28154841/153408
if __name__ == "__main__" and __package__ is None:
    __package__ = "propagation"
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from . import mpi, optimize, propagation, read, simulation


def set_excepthook():
    oldhook = sys.excepthook

    def newhook(*args, **kwargs):
        oldhook(*args, **kwargs)
        MPI.COMM_WORLD.Abort(1)

    sys.excepthook = newhook


def pd_setup():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--runid')
    parser.add_argument('--graph', help="graph file (.npz, .metis or .adjlist)")
    parser.add_argument('--metis_zero_based', action='store_true')
    parser.add_argument('--tweets', help="tweet file (.csv)")
    parser.add_argument('--stats', help="stats file (.csv)")
    parser.add_argument('--params', help="params file (.csv)")
    parser.add_argument('--discount', help="discount file (.csv)")
    parser.add_argument('--corr', help="corr file (.csv)")
    parser.add_argument('--source_map', help="source map file (.csv)")
    parser.add_argument('--indir', help="input directory (default: data)", default='data')
    parser.add_argument('--outdir', help="output directory (default: out)", default='out')
    parser.add_argument(
        '-f', '--features', help="number of features to simulate", type=int, default=1
    )
    parser.add_argument(
        '-a', '--sources', help="number of authors per feature class", type=int, default=1
    )
    parser.add_argument('-s', '--samples', help="number of samples per tweet", type=int, default=1)
    parser.add_argument(
        '--epsilon', help="epsilon for parameter learning", type=float, default=0.001
    )
    parser.add_argument('--max_depth', help="maximum depth to simulate", type=int)
    parser.add_argument('--max_nodes', help="maximum retweet count to simulate", type=int)
    parser.add_argument(
        '--sample_split',
        help="split tasks (default: 1, has to divide 'samples', changes RNG)",
        type=int,
        default=1,
    )
    parser.add_argument('--seed', help="seed for RNG", type=int)
    parser.add_argument(
        "command",
        choices=['learn_discount', 'learn_corr', 'optimize', 'sim', 'simtweets', 'val'],
    )
    parser.add_argument("topic")
    args = parser.parse_args()

    # Defaults
    if args.runid is None:
        if 'RUNID' in os.environ:
            args.runid = os.environ.get('RUNID')
        elif 'PBS_JOBID' in os.environ:
            args.runid = os.environ.get('PBS_JOBID')
        elif 'SLURM_JOB_ID' in os.environ:
            args.runid = os.environ.get('SLURM_JOB_ID')
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
    """Construct simulation object."""
    if args.graph.endswith('.adjlist'):
        A, node_labels = read.adjlist(
            args.graph, save_as=args.graph.replace('.adjlist', '.adjlist.npz')
        )
    elif args.graph.endswith('.metis'):
        A, node_labels = read.metis(
            args.graph,
            zero_based=args.metis_zero_based,
            save_as=args.graph.replace('.metis', '.npz'),
        )
    elif args.graph.endswith('.npz'):
        A, node_labels = read.labelled_graph(args.graph)
        if args.tweets:
            print('Assuming METIS style ids.')
    else:
        raise ValueError(f"Unknown graph file format {args.graph} (use .metis, .npz or .adjlist).")

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

    params = read.params(args.params) if args.params else None
    sim = simulation.Simulation(A, stats, source_map, params=params, seed=args.seed)

    if args.max_depth:
        sim.params.max_depth = args.max_depth
    if args.max_nodes:
        sim.params.max_nodes = args.max_nodes

    if args.discount:
        sim.params['discount_factor'] = read.single_param(args.discount)

    if args.corr:
        sim.params['corr'] = read.single_param(args.corr)

    # Sort features by decreasing expected runtime
    sim.stats.sort_values(
        by=['mean_retweets', 'retweet_probability'], ascending=False, inplace=True
    )
    sim.features = sim.stats.index  # TODO check if necessary
    sim.params = sim.params.reindex(index=sim.features)

    return sim


def setup():
    set_excepthook()
    pd_setup()

    args = parse_args()

    print("date:", datetime.now().isoformat())
    print("mpi_size:", MPI.COMM_WORLD.Get_size())
    print("mpi_vendor:", MPI.get_vendor())
    code_version = subprocess.run(
        ['git', 'describe', '--tags', '--dirty'], capture_output=True, text=True
    ).stdout.strip()
    print("code_version:", code_version)
    print("argv:", ' '.join(sys.argv))
    print("args:", vars(args))

    sim = build_sim(args)

    print("seed:", sim.seed.entropy)
    return sim, args


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
    return r.groupby(['author_feature', 'tweet_feature']).agg(
        tweets=('feature', 'size'),  # TODO: multiply with authors * samples
        mean_retweets=('mean_retweets', 'mean'),
        retweet_probability=('retweet_probability', 'mean'),
    )


def explode_tweets(tweet_results):
    r = pd.DataFrame(tweet_results, columns=['feature', 'results'])
    r[['author_feature', 'tweet_feature']] = pd.DataFrame(r['feature'].tolist())

    r['results'] = r['results'].apply(list)
    r = r.explode('results', ignore_index=True)

    r[['author', 'retweets']] = pd.DataFrame(r['results'].tolist())

    r['retweets'] = r['retweets'].apply(list)
    r = r.explode('retweets', ignore_index=True)

    return r[['author', 'author_feature', 'tweet_feature', 'retweets']]


def mae(sim, real=0.0):
    return (sim - real).abs().mean()


def mape(sim, real):
    return ((sim - real) / real).abs().mean()


def wmape(sim, real):
    return ((sim - real) / real.mean()).abs().mean()


def run(sim, args):
    if args.command == 'learn_discount':
        discount = optimize.discount_from_mean_retweets(sim, samples=args.samples, eps=args.epsilon)
        discount.to_csv(f'{args.outdir}/discount-{args.topic}-{args.runid}.csv')
        sim.params['discount_factor'] = discount
        sim.params.to_csv(f'{args.outdir}/params-{args.topic}-{args.runid}.csv')

    elif args.command == 'learn_corr':
        corr = optimize.corr_from_mean_retweets(sim, samples=args.samples, eps=args.epsilon)
        corr.to_csv(f'{args.outdir}/corr-{args.topic}-{args.runid}.csv')
        sim.params['corr'] = corr
        sim.params.to_csv(f'{args.outdir}/params-{args.topic}-{args.runid}.csv')

    elif args.command == 'optimize':
        best, state = optimize.gridsearch(
            sim,
            # num=1,
            sources=None if args.sources < 1 else args.sources,
            samples=args.samples,
            eps=args.epsilon,
        )
        optimize.set_params(best, sim)
        sim.params.to_csv(f'{args.outdir}/params-{args.topic}-{args.runid}.csv')
        with open(f'{args.outdir}/optimize-{args.topic}-{args.runid}.pickle', 'bw') as f:
            pickle.dump((best, state), f)
        # last history element in first optimization
        # objective = pd.Series({k: o[0][1][-1] for k, o in opts.items()})
        # real = sim.stats.mean_retweets
        # sim = real + objective
        # print(f'mae: {mae(sim, real)}')
        # print(f'mape: {mape(sim, real)}')
        # print(f'wmape: {wmape(sim, real)}')

    elif args.command == 'val':
        print(f'{len(sim.features)} features, {args.sources} sources, {args.samples} samples')
        r = agg_statistics(
            (feature, sim.simulate(feature, sources=args.sources, samples=args.samples))
            for feature in sim.features
        )
        # for feature in [('0000', '0000')])

        # assert r.index.equals(sim.stats.index)
        # r = r.reindex(index=sim.features)
        r.columns = pd.MultiIndex.from_product([['sim'], r.columns])
        r[('real', 'mean_retweets')] = sim.stats.mean_retweets
        r[('real', 'retweet_probability')] = sim.stats.retweet_probability

        r.to_csv(f'{args.outdir}/results-val-{args.topic}-{args.runid}.csv')
        # r = pd.read_csv(..., header=[0,1], index_col=[0,1])

        pretty = r.swaplevel(axis=1).sort_index(axis=1).drop(columns='tweets')
        pretty.index = r.index.to_flat_index()
        print(pretty)

        def print_error(measure, stat, real=r.real, sim=r.sim):
            print(f"{measure.__name__}_{stat}: {measure(sim[stat], real[stat])}")

        for stat in ['retweet_probability', 'mean_retweets']:
            for measure in [mae, mape, wmape]:
                print_error(measure, stat)

    elif args.command == 'sim':
        r = agg_statistics(
            (feature, sim.simulate(feature, sources=args.sources, samples=args.samples))
            for feature in sim.sample_feature(args.features)
        )
        r.to_csv(f'{args.outdir}/results-sim-{args.topic}-{args.runid}.csv')
        print(r)

    elif args.command == 'simtweets':
        r = explode_tweets(
            sim.run(num_features=args.features, num_sources=args.sources, samples=args.samples)
        )
        r.to_csv(f'{args.outdir}/results-simtweets-{args.topic}-{args.runid}.csv')
        print(r)


def main():
    is_head = MPI.COMM_WORLD.Get_rank() == 0

    if is_head:
        start_time = time.time()
        sim, args = setup()
        t = time.time()
        print("readtime:", t - start_time, flush=True)
    else:
        propagation.compile()
        sim = None

    # if True: # bypass mpi
    with mpi.futures(
        sim,
        chunksize=1,
        sample_split=args.sample_split if is_head else None,
        fixed_samples=args.samples if is_head else None,
    ) as sim:
        if sim is not None:
            assert is_head
            print("setuptime:", time.time() - t, flush=True)
            t = time.time()

            run(sim, args)

            print("runtime:", time.time() - t)
            print("totaltime:", time.time() - start_time)
            print("ru_self:", resource.getrusage(resource.RUSAGE_SELF))
            print("ru_children:", resource.getrusage(resource.RUSAGE_CHILDREN), flush=True)
            if hasattr(resource, 'RUSAGE_BOTH'):
                print("ru_both:", resource.getrusage(resource.RUSAGE_BOTH), flush=True)
            MPI.COMM_WORLD.Abort(0)


if __name__ == "__main__":
    main()
