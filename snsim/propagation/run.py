#!/usr/bin/env python
import argparse
import os
import resource
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
from mpi4py import MPI

from . import commands, mpi, propagation, read, simulation


def set_excepthook():
    oldhook = sys.excepthook

    def newhook(*args, **kwargs):
        oldhook(*args, **kwargs)
        sys.stdout.flush()
        MPI.COMM_WORLD.Abort(1)

    sys.excepthook = newhook


def pd_setup():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument('--runid')
    p.add_argument('--graph', help="graph file (.npz, .metis or .adjlist)")
    p.add_argument('--metis_zero_based', action='store_true')
    p.add_argument('--tweets', help="tweet file (.csv)")
    p.add_argument('--stats', help="stats file (.csv)")
    p.add_argument('--params', help="params file (.csv)")
    p.add_argument('--discount', help="discount file (.csv)")
    p.add_argument('--corr', help="corr file (.csv)")
    p.add_argument('--source_map', help="source map file (.csv)")
    p.add_argument('--indir', help="input directory (default: data)", default='data')
    p.add_argument('--outdir', help="output directory (default: out)", default='out')
    p.add_argument(
        '-f', '--features', help="number of features to simulate (default: 1)", type=int, default=1
    )
    p.add_argument(
        '-a',
        '--sources',
        help="number of authors per feature class (default: 1)",
        type=int,
        default=1,
    )
    p.add_argument(
        '-s', '--samples', help="number of samples per tweet (default: 1)", type=int, default=1
    )
    p.add_argument(
        '--epsilon',
        help="epsilon for parameter learning (default: 0.001)",
        type=float,
        default=0.001,
    )
    p.add_argument(
        '--steps',
        help="number of steps for optimization (default: 10)",
        type=int,
        default=10,
    )
    p.add_argument('--max_depth', help="maximum depth to simulate", type=int)
    p.add_argument('--max_nodes', help="maximum retweet count to simulate", type=int)
    p.add_argument(
        '--sample_split',
        help="split tasks (default: 1, has to divide 'samples', changes RNG)",
        type=int,
        default=1,
    )
    p.add_argument('--seed', help="seed for RNG", type=int)
    p.add_argument('--split', help="", default=1, type=int)
    p.add_argument(
        "command",
        choices=commands.cmds.keys(),
    )
    p.add_argument("topic")

    args = p.parse_args()

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
    features_sorted = sim.stats.sort_values(
        by=['mean_retweets', 'retweet_probability'], ascending=False
    ).index
    sim.reindex(features_sorted)

    return sim


def setup():
    set_excepthook()
    pd_setup()

    args = parse_args()

    print("date:", datetime.now().isoformat())
    print("mpi_size:", MPI.COMM_WORLD.Get_size())
    print("mpi_vendor:", MPI.get_vendor())
    if sys.version >= '3.7':
        code_version = subprocess.run(
            ['git', 'describe', '--tags', '--dirty'], capture_output=True, text=True
        ).stdout.strip()
        print("code_version:", code_version)
    print("argv:", ' '.join(sys.argv))
    print("args:", vars(args))

    sim = build_sim(args)

    print("seed:", sim.seed.entropy)
    return sim, args


def rusage():
    keys = (
        'utime',
        'stime',
        'maxrss',
        'ixrss',
        'idrss',
        'isrss',
        'minflt',
        'majflt',
        'nswap',
        'inblock',
        'oublock',
        'msgsnd',
        'msgrcv',
        'nsignals',
        'nvcsw',
        'nivcsw',
    )
    return dict(zip(keys, resource.getrusage(resource.RUSAGE_SELF)))


def main():
    is_head = MPI.COMM_WORLD.Get_rank() == 0

    if is_head:
        start_time = time.time()
        sim, args = setup()
        print(f'{len(sim.features)} features, {args.sources} sources, {args.samples} samples')
        t = time.time()
        print("readtime:", t - start_time, flush=True)
        # mpi_sim = mpi.futures(sim=sim, sample_split=args.sample_split, fixed_samples=args.samples)
        split = mpi.split(
            sim=sim, args=args, sample_split=args.sample_split, fixed_samples=args.samples
        )
    else:
        propagation.compile()
        split = mpi.split()

    with split as (args, sim):
        if sim:
            if is_head:
                print("setuptime:", time.time() - t, flush=True)
                t = time.time()
            commands.run(sim, args)

    if is_head:
        print("runtime:", time.time() - t)
        print("totaltime:", time.time() - start_time)
        print("rusage:", rusage(), flush=True)
        # MPI.COMM_WORLD.Abort(0)


if __name__ == "__main__":
    main()
