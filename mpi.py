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


def distribute_jobs(jobs, func, comm=MPI.COMM_WORLD, head=0):
    rank = comm.Get_rank()
    num_worker = comm.Get_size() - 1
    assert len(jobs) >= num_worker

    if rank == head:
        results = [None] * len(jobs)
        status = MPI.Status()
        for next_job in range(num_worker, num_worker + len(jobs)):
            result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            comm.send(next_job, status.Get_source())
            results[status.Get_tag()] = result
        return results
    else:
        job = rank - 1
        while job < len(jobs):
            result = func(jobs[job])
            job = comm.sendrecv(result, head, sendtag=job)
        return None


def split_jobs(sources, runs_per_source, num_chunks):
    num_runs = len(sources) * runs_per_source
    batch_size, r = divmod(num_runs, num_chunks)
    if r > 0:
        batch_size += 1

    jobs = []
    current_source = 0
    current_remaining = runs_per_source
    for _ in range(num_chunks):
        quota = batch_size
        subjobs = []
        while quota > 0:
            if current_source >= len(sources):
                break
            if current_remaining <= quota:
                subjobs.append((sources[current_source], current_remaining))
                quota -= current_remaining
                current_source += 1
                current_remaining = runs_per_source
            else:
                subjobs.append((sources[current_source], quota))
                current_remaining -= quota
                quota = 0
        jobs.append(subjobs)
    assert current_source == len(sources)

    return jobs


def mpi_simulator(A, comm=MPI.COMM_WORLD, root=0, num_chunks=None):
    """
    Returns: simulate function (that ignores its first argument).
    """
    rank = comm.Get_rank()
    assert A is not None or rank != 0
    size = comm.Get_size()

    assert root == 0
    A_global = bcast_csr_matrix(A, comm)

    if num_chunks is None:
        num_chunks = size * 4

    def simulate(A: None, sources, p, corr, discount=1., depth=None, max_nodes=None, samples=1,
                 return_stats=True):
        """Simulate tweets starting from sources, return mean retweets and retweet probability."""
        assert return_stats == True
        jobs = split_jobs(sources, samples, num_chunks)
        if rank == root:
            print(jobs)

        def s(subjobs):
            retweets = (
                (propagation.edge_propagate(A_global, source, p=p, corr=corr, discount=discount, depth=depth,
                                            max_nodes=max_nodes) for _ in range(samples))
                for source, samples in subjobs)
            retweets = [tweet for source in retweets for tweet in source]  # Flatten
            return sum(retweets), np.count_nonzero(retweets)

        results = distribute_jobs(jobs, s)

        if rank == root:
            num_runs = len(sources) * samples
            mean = sum(x for x, _ in results)
            prob = sum(x for _, x in results)
            return mean / num_runs, prob / num_runs

    return simulate


global_A = None


def worker(params):
    # print(params)
    return propagation.simulate(global_A, *params)


@contextmanager
def futures(sim, comm=MPI.COMM_WORLD, root=0, num_chunks=None):
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
    def simulate(A: None, sources, p, corr=0., discount=1., depth=None, max_nodes=None, samples=1, return_stats=True):
        """Simulate tweets starting from sources, return mean retweets and retweet probability."""
        samples = max(samples // num_chunks, 1)
        print(f'{samples * num_chunks} samples')

        sample_calls = [(sources, p, corr, discount, depth, max_nodes, samples, return_stats) for _ in
                        range(num_chunks)]
        results = executor.map(worker, sample_calls)
        if return_stats:
            return stats_from_futures(results)

        assert False
        return results

    rank = comm.Get_rank()
    A = None
    if rank == 0:
        A = sim.A

    size = comm.Get_size()

    if num_chunks is None:
        num_chunks = size * 4

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
    mean_retweets, retweet_probability = tuple(zip(*results))
    return np.mean(mean_retweets), np.mean(retweet_probability)
