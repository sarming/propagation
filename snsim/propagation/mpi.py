from contextlib import contextmanager
from itertools import chain, islice

import numpy as np
import pandas as pd
from mpi4py import MPI
from scipy.sparse import csr_matrix

from . import propagation


def bcast_array(arr=None, comm=MPI.COMM_WORLD, root=0):
    """Broadcast an arbitrary numpy array."""

    rank = comm.Get_rank()
    assert arr is not None or rank != root

    if rank == root:
        shape = arr.shape
        dtype = arr.dtype
        comm.bcast((shape, dtype), root=root)
    else:
        shape, dtype = comm.bcast(None, root=root)
        arr = np.empty(shape=shape, dtype=dtype)
    comm.Bcast(arr, root=root)
    return arr


def bcast_array_shm(arr=None, comm=MPI.COMM_WORLD, root=0):
    """Broadcast a numpy array via shared memory.

    Args:
        arr: numpy array
        comm: shared memory communicator
        root: root rank in comm

    Returns: copy of arr in shared memory
    """
    rank = comm.Get_rank()
    assert arr is not None or rank != root

    if rank == root:
        shape = arr.shape
        dtype = arr.dtype
        nbytes = arr.nbytes
        comm.bcast((shape, dtype, nbytes), root=root)
    else:
        (shape, dtype, nbytes) = comm.bcast(None, root=root)
    win = MPI.Win.Allocate_shared(nbytes if rank == 0 else 0, MPI.BYTE.Get_size(), comm=comm)
    buf, _ = win.Shared_query(root)
    new_arr = np.ndarray(buffer=buf, dtype=dtype, shape=shape)
    if rank == root:
        np.copyto(new_arr, arr)
    comm.Barrier()
    return new_arr


def bcast_csr_matrix(A=None, comm=MPI.COMM_WORLD):
    """Broadcast a csr_matrix to shared memory of every node.

    Args:
        A: csr_matrix (must be set in rank 0)
        comm: MPI communicator

    Returns: copy of A in shared memory

    """
    rank = comm.Get_rank()
    assert A is not None or rank != 0

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    # head_comm = comm.Split(True if node_rank == 0 else MPI.UNDEFINED)
    head_comm = comm.Split(node_rank)

    Ad = Ai = Ap = None
    if rank == 0:
        Ad = A.data
        Ai = A.indices
        Ap = A.indptr
    if node_rank == 0:  # or rank == 0
        Ad = bcast_array(Ad, head_comm)
        Ai = bcast_array(Ai, head_comm)
        Ap = bcast_array(Ap, head_comm)

    Ad = bcast_array_shm(Ad, node_comm)
    Ai = bcast_array_shm(Ai, node_comm)
    Ap = bcast_array_shm(Ap, node_comm)

    return csr_matrix((Ad, Ai, Ap))


def bcast_sim(sim, comm, root=0):
    rank = comm.Get_rank()
    if rank == root:
        A = sim.A
        seed = [s.state for s in sim.seed.spawn(comm.Get_size())]
        sim.A = None
        sim.seed = None
        sim.rng = None
    else:
        sim = None
        A = None
        seed = None

    sim = comm.bcast(sim, root=root)
    assert root == 0
    A = bcast_csr_matrix(A, comm)
    seed = comm.scatter(seed, root=root)

    sim.A = A
    sim.seed = np.random.SeedSequence(**seed)
    sim.rng = np.random.default_rng(sim.seed)

    comm.Barrier()
    return sim


@contextmanager
def split(
    n_splits=2, sim=None, comm=MPI.COMM_WORLD, root=0, args={}, *futures_args, **futures_kwargs
):
    assert comm.Get_size() % n_splits == 0

    global_rank = comm.Get_rank()
    split_comm = comm.Split(global_rank % n_splits)
    split_rank = split_comm.Get_rank()
    split_root = root % n_splits
    assert split_rank == global_rank // n_splits

    heads = comm.Split(split_rank)
    if split_rank == split_root:
        sim = bcast_sim(sim, heads, root)
        args = heads.bcast(args, root)
        futures_args = heads.bcast(futures_args, root)
        futures_kwargs = heads.bcast(futures_kwargs, root)
        assert sim is not None

        head_rank = heads.Get_rank()
        args.split = head_rank

        features = sim.features.to_list()
        features = features[head_rank::n_splits]
        features = pd.MultiIndex.from_tuples(features, names=("author_feature", "tweet_feature"))
        sim.reindex(features)

    with futures(sim, comm=split_comm, root=split_root, *futures_args, **futures_kwargs) as sim:
        yield args, sim


global_A = None
global_entropy = None
global_samples = None
global_params = None


def worker(params, sources, spawn_key, samples=None, return_stats=True):
    """Worker function for futures implentation below."""
    params = dict(zip(global_params, params))
    # print(params, flush=True)
    if samples is None:
        samples = global_samples
    seed = np.random.SeedSequence(entropy=global_entropy, spawn_key=spawn_key)
    r = propagation.simulate(global_A, params, [sources], samples, return_stats, seed)

    if return_stats:
        return r
    else:
        return list(list(r)[0])
        # return [list(source) for source in r]


def worker_return_tweets(*args):
    return worker(*args, return_stats=False)


@contextmanager
def futures(sim=None, comm=MPI.COMM_WORLD, root=0, chunksize=1, sample_split=1, fixed_samples=None):
    from mpi4py.futures import MPICommExecutor

    if comm.Get_size() == 1:
        yield sim
        return

    # @timecall
    def simulate(A: None, sources, params, samples=1, return_stats=True, seed=None):
        """Simulate tweets starting from sources, return mean retweets and retweet probability."""
        seeds = seed.spawn(len(sources) * sample_split)
        sources = sources * sample_split
        params = list(dict(params).values())

        sample_calls = (
            (params, source, seed.state['spawn_key'])
            if global_samples and global_samples == samples // sample_split
            else (params, source, seed.state['spawn_key'], samples // sample_split)
            for source, seed in zip(sources, seeds)
        )

        w = worker if return_stats else worker_return_tweets
        results = executor.starmap(w, sample_calls, chunksize=chunksize, unordered=return_stats)

        if return_stats:
            return stats_from_futures(results)
        return (chain.from_iterable(islice(results, sample_split)) for _ in sources)

    global global_A
    global global_entropy
    global global_samples
    global global_params

    assert root == 0
    assert global_A is None

    rank = comm.Get_rank()

    if rank == 0:
        global_A = sim.A
        global_entropy = sim.seed.state['entropy']
        if fixed_samples:
            assert fixed_samples % sample_split == 0
            global_samples = fixed_samples // sample_split
        global_params = list(dict(sim.params).keys())

    global_A = bcast_csr_matrix(global_A, comm)
    global_entropy = comm.bcast(global_entropy, root=root)
    global_samples = comm.bcast(global_samples, root=root)
    global_params = comm.bcast(global_params, root=root)

    MPI.COMM_WORLD.Barrier()

    with MPICommExecutor(comm=comm, root=root) as executor:
        if executor is None:
            yield None
        else:
            old_simulator = sim.simulator
            sim.simulator = simulate
            yield sim
            executor.shutdown(wait=False)
            sim.simulator = old_simulator

    global_A = None


def stats_from_futures(results):
    """Lazily (as generator instead of pair) compute mean_retweets and retweet_probability.

    Args:
        results: iterable of (mean_retweets, retweet_probability) pairs.

    Yields: mean mean_retweets followed by mean retweet_probability.

    """
    results = list(results)
    # print(results)
    mean_retweets, retweet_probability = tuple(zip(*results))  # list of pairs to pair of lists
    # return np.mean(mean_retweets), np.mean(retweet_probability)
    yield np.mean(mean_retweets)
    yield np.mean(retweet_probability)
