from contextlib import contextmanager

import numpy as np
import scipy as sp
from mpi4py import MPI

import propagation


def bcast_array(arr=None, comm=MPI.COMM_WORLD, root=0):
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
    """comm needs to be shared memory."""
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
    rank = comm.Get_rank()
    assert A is not None or rank != 0

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    head_comm = comm.Split(True if node_rank == 0 else MPI.UNDEFINED)

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

    return sp.sparse.csr_matrix((Ad, Ai, Ap))


global_A = None


def worker(args):
    # print(params)
    r = propagation.simulate(global_A, *args)

    if args[3]:  # return_stats
        return r
    else:
        return list(list(r)[0])
        # return [list(source) for source in r]


@contextmanager
def futures(sim, comm=MPI.COMM_WORLD, root=0, chunksize=1):
    """Return a simulate function using mpi4py.futures for a fixed matrix A.
    FIXME

    Note:
        The graph must be passed up front to this function.
        The returned function ignores its first argument.

    Args:
        A: matrix used for all simulate calls

    Returns: simulate function (that ignores its first argument).
    """
    from mpi4py.futures import MPICommExecutor

    # @timecall
    def simulate(A: None, sources, params, samples=1, return_stats=True):
        """Simulate tweets starting from sources, return mean retweets and retweet probability."""
        sample_calls = [([source], params, samples, return_stats) for source in sources]
        results = executor.map(worker, sample_calls, chunksize=chunksize)

        if return_stats:
            return stats_from_futures(results)
        return results

    rank = comm.Get_rank()
    A = None
    if rank == 0:
        A = sim.A

    size = comm.Get_size()

    assert root == 0

    global global_A
    assert global_A is None
    global_A = bcast_csr_matrix(A, comm)

    with MPICommExecutor(comm=comm, root=root) as executor:
        if executor is None:
            yield None
        else:
            old_simulator = sim.simulator
            sim.simulator = simulate
            yield sim
            sim.simulator = old_simulator

    global_A = None


def stats_from_futures(results):
    results = list(results)
    # print(results)
    mean_retweets, retweet_probability = tuple(zip(*results))  # list of pairs to pair of lists
    # return np.mean(mean_retweets), np.mean(retweet_probability)
    yield np.mean(mean_retweets)
    yield np.mean(retweet_probability)
