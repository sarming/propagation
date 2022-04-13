import numpy as np
from numba import njit


def edge_propagate(
    A, source, p, corr=0.0, discount=1.0, depth=None, max_nodes=None, at_least_one=True
):
    """Propagate message in graph A and return number of nodes visited.

    Args:
        A: Sparse adjacency matrix of graph.
        source (int): Initial node.
        p (float): Probability parameter.
        corr (float): Correlation probability.
        discount (float): Discount factor <=1.0 that is multiplied at each level.
        depth (int): Maximum depth.
        max_nodes (int): Maximum number of nodes.
        at_least_one (bool): If true, p is retweet probability. If false, p is edge probability.

    Returns:
        Number of nodes visited (without initial).

    """
    # print(f"propagate from {source}")
    # return edge_propagate_tree(A, start, p, discount, depth).number_of_nodes() - 1
    if depth is None:
        depth = int(A.shape[0])
    if max_nodes is None:
        max_nodes = int(A.shape[0])
    visited = {source}
    leaves = {source}
    # done = {source}
    for _ in range(depth):
        next_leaves = set()
        for node in leaves:
            children = set(edge_sample(A, node, p, corr, at_least_one))
            children -= visited
            next_leaves |= children
            visited |= children
            # done |= set(A.indices[A.indptr[node]:A.indptr[node + 1]])
            if len(visited) > max_nodes:
                return max_nodes
        leaves = next_leaves
        p *= discount
        if not leaves:
            break
    # print(f"done {len(visited)}")
    return len(visited) - 1


def edge_propagate_tree(
    A, source, p, corr=0.0, discount=1.0, depth=None, max_nodes=None, at_least_one=True
):
    """Propagate message in graph A and return retweet tree.

    Args:
        A: Sparse adjacency matrix of graph.
        source (int): Initial node.
        p (float): Probability parameter.
        corr (float): Correlation probability.
        discount (float): Discount factor <=1.0 that is multiplied at each level.
        depth (int): Maximum depth.
        max_nodes (int): Maximum number of nodes.
        at_least_one (bool): If true, p is retweet probability. If false, p is edge probability.

    Returns:
        Retweet tree as dict.

    """
    if depth is None:
        depth = int(A.shape[0])
    if max_nodes is None:
        max_nodes = int(A.shape[0])
    visited = {source}
    leaves = {source}
    tree = {source: np.int32(-1)}
    for _ in range(depth):
        next_leaves = set()
        for node in leaves:
            children = set(edge_sample(A, node, p, corr, at_least_one))
            children -= visited
            next_leaves |= children
            visited |= children
            for c in children:
                tree[c] = node
            if len(visited) > max_nodes:
                return tree
        leaves = next_leaves
        p *= discount
    return tree


def edge_sample(A, node, p, corr=0.0, at_least_one=True):
    """Return sample of node's children using probability p.

    Note:
         This is the inner loop, rewrite in Cython might be worthwhile.
    """
    return edge_sample_numba(A.indptr, A.indices, node, p, corr, at_least_one)


@njit
def edge_sample_numba(indptr, indices, node, p, corr, at_least_one):
    l, r = indptr[node], indptr[node + 1]
    num_follower = r - l
    if num_follower == 0:
        return [np.int32(x) for x in range(0)]
        # return np.empty(1, dtype=np.int32)
    if at_least_one:
        p = 1 - (1 - p) ** (1 / num_follower)
    num_retweeter = np.random.binomial(num_follower, p)
    if num_retweeter > 0 and corr > 0:
        num_retweeter += np.random.binomial(num_follower - num_retweeter, corr)

    children = indices[l:r]
    return list(np.random.choice(children, num_retweeter, replace=False))


@njit
def set_seed(seed):
    # print(seed)
    np.random.seed(seed)


def compile():
    from scipy.sparse import csr_matrix

    edge_sample(csr_matrix([[1, 1], [1, 1]]), 0, 0.0)


def simulation_stats(simulation_results):
    """Take simulation results (list of lists) and return mean retweets and retweet probability."""
    retweets = [tweet for source in simulation_results for tweet in source]  # Flatten
    return sum(retweets) / len(retweets), np.count_nonzero(retweets) / len(retweets)


# @timecall
def simulate(A, params, sources, samples=1, return_stats=True, seed=None):
    """Propagate messages and return mean retweets and retweet probability.

    Args:
        A: Sparse adjacency matrix of graph.
        sources (list): List of source nodes.
        params (dict-like): Simulation parameters.
        samples (int): Number of samples per source node.
        return_stats (bool): If set to false, return full results (list of lists) instead of stats.

    Returns:
        (int, int): Mean retweets and retweet probability over all runs.
    """
    # print('.', end='', flush=True)
    p = params['edge_probability']
    corr = params['corr']
    depth = params['max_depth']
    max_nodes = params['max_nodes']
    at_least_one = params['at_least_one']
    discount = params['discount_factor']

    if isinstance(seed, dict):
        seed = np.random.SeedSequence(**seed)
    set_seed(seed.generate_state(1)[0])

    retweets = (
        [
            edge_propagate(
                A,
                source,
                p=p,
                corr=corr,
                discount=discount,
                depth=depth,
                max_nodes=max_nodes,
                at_least_one=at_least_one,
            )
            for _ in range(samples)
        ]
        for source in sources
    )
    if return_stats:
        return simulation_stats(retweets)
    return retweets
