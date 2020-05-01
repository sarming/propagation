import itertools
import multiprocessing

import ray
import scipy as sp

import propagation


def ray_simulator(num_chunks=None):
    @ray.remote
    def worker(A, sources, p, discount, depth, max_nodes, samples):
        A = sp.sparse.csr_matrix(*A)
        return [
            [propagation.edge_propagate(A, source, p=p, discount=discount, depth=depth, max_nodes=max_nodes) for _ in
             range(samples)] for source in sources]

    def chunk(lst, n):
        """Split lst into about n chunks of equal size."""
        s, r = divmod(len(lst), n)
        if r:
            s += 1
        for i in range(0, len(lst), s):
            yield lst[i:i + s]

    def simulate(A, sources, p, discount=1., depth=1, max_nodes=1000, samples=1, return_stats=True):
        """Simulate tweets starting from sources, return mean retweets and retweet probability."""
        A = ray.put(((A.data, A.indices, A.indptr), A.shape))
        res = [worker.remote(A, s, p, discount, depth, max_nodes, samples) for s in chunk(sources, num_chunks)]
        retweets = itertools.chain(*ray.get(res))

        if return_stats:
            return propagation.simulation_stats(retweets)
        return retweets

    ray.init(ignore_reinit_error=True)
    if num_chunks is None:
        num_chunks = int(ray.cluster_resources()['CPU']) * 4

    return simulate


def make_global(A):
    global global_A
    global_A = A


def pool_worker(source, p, discount, depth, max_nodes, samples):
    return [propagation.edge_propagate(global_A, source, p=p, discount=discount, depth=depth, max_nodes=max_nodes)
            for _ in range(samples)]


def pool_simulator(A, processes=None):
    def simulate(A: None, sources, p, discount=1., depth=1, max_nodes=1000, samples=1, return_stats=True):
        """Simulate tweets starting from sources, return mean retweets and retweet probability."""
        sample_calls = [(source, p, discount, depth, max_nodes, samples) for source in sources]
        retweets = pool.starmap(pool_worker, sample_calls)
        if return_stats:
            return propagation.simulation_stats(retweets)
        return retweets

    pool = multiprocessing.Pool(processes, initializer=make_global, initargs=(A,))
    return simulate
