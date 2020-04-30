import numpy as np
import pandas as pd
from profilehooks import timecall

import parallel
import read


def edge_propagate(A, start, p, discount=1., depth=1, max_nodes=1000):
    """Propagate message in graph A and return number of nodes visited.

    Args:
        A: Sparse adjacency matrix of graph.
        start (int): Initial node.
        p (float): Probability that message passes along an edge.
        discount (float): Discount factor <=1.0 that is multiplied at each level.
        depth (int): Maximum depth.
        max_nodes (int): Maximum number of nodes.

    Returns:
        Number of nodes visited (without initial).

    """
    # return edge_propagate(A, start, p, discount, depth).number_of_nodes() - 1
    visited = {start}
    leaves = {start}
    for i in range(depth):
        next_leaves = set()
        for node in leaves:
            children = set(edge_sample(A, node, p * discount ** i))
            children -= visited
            next_leaves |= children
            visited |= children
            if len(visited) > max_nodes:
                return max_nodes
        leaves = next_leaves
    return len(visited) - 1


def edge_sample(A, node, p):
    """Return Bernoulli sample of node's children using probability p.

    Note:
         This is the inner loop, rewrite in Cython might be worthwhile.
    """
    l, r = A.indptr[node], A.indptr[node + 1]
    # return A.indices[l:r][np.random.rand(r - l) < p]
    if l == r:
        return []
    num = np.random.binomial(r - l, p)

    # return A.indices[np.random.choice(r - l, num, replace=False) + l]
    children = A.indices[l:r]
    return np.random.choice(children, num, replace=False)


def calculate_retweet_probability(A, sources, p):
    """Return average number of retweeted messages when starting from sources using edge probability p.

    Args:
        A: Adjacency matrix of graph.
        sources: List of source nodes, one per tweet.
        p: Edge probability.

    Returns: mean_{x in sources} 1-(1-p)^{deg-(x)}
    """
    return sum(1 - (1 - p) ** float(A.indptr[x + 1] - A.indptr[x]) for x in sources) / len(sources)


def invert_monotone(fun, goal, lb, ub, eps):
    """Find fun^-1( goal ) by binary search.

    Note:
        For correctness fun has to be monotone up to eps, viz. fun(x) <= fun(x+eps) for all x
        This is an issue with stochastic functions.

    Args:
        fun: Monotone Function.
        goal: Goal value.
        lb: Initial lower bound.
        ub: Initial upper bound.
        eps: Precision at which to stop.

    Returns: x s.t. lb<x<ub and there is y with |x-y|<=eps and fun(y)=goal
    """
    mid = (ub + lb) / 2
    if ub - lb < eps:
        return mid
    f = fun(mid)
    print(f"f({mid})={f}{'<' if f < goal else '>'}{goal} [{lb},{ub}]")
    if f < goal:
        return invert_monotone(fun, goal, mid, ub, eps)
    else:
        return invert_monotone(fun, goal, lb, mid, eps)


def simulation_stats(simulation_results):
    retweets = [tweet for source in simulation_results for tweet in source]  # Flatten
    return sum(retweets) / len(retweets), np.count_nonzero(retweets) / len(retweets)


# @timecall
def simulate(A, sources, p, discount=1., depth=1, max_nodes=1000, samples=1, return_stats=True):
    """Simulate tweets starting from sources, return mean retweets and retweet probability."""
    retweets = ((edge_propagate(A, source, p=p, discount=discount, depth=depth, max_nodes=max_nodes)
                 for _ in range(samples)) for source in sources)
    if return_stats:
        return simulation_stats(retweets)
    return retweets


def logging(f):
    def wrapped(*args, **kwargs):
        print(f'calling {f.__name__}({args},{kwargs})')
        return f(*args, **kwargs)

    return wrapped


class Simulation:
    def __init__(self, A, tweets):
        self.A = A
        self.stats = Simulation.tweet_statistics(tweets)
        self.params = pd.DataFrame({'freq': self.stats.tweets / self.stats.tweets.sum(),
                                    'edge_probability': np.NaN,  # will be calculated below
                                    'discount_factor': 1.0,
                                    'max_retweets': 10 * self.stats.max_retweets,
                                    'depth': 10,
                                    })
        self.sources = tweets.dropna().groupby('author_feature')['source'].unique()
        self.features = self.stats.index

        self._sim = simulate
        # self._sim = parallel.ray_simulator(8)
        # self._sim = parallel.pool_simulator(self.A)
        self._sim = logging(self._sim)

        self.params['edge_probability'] = self.edge_probability_from_retweet_probability()

    @staticmethod
    def tweet_statistics(tweets, min_size=10):
        stats = tweets.groupby(['author_feature', 'tweet_feature']).agg(
            tweets=('source', 'size'),
            retweet_probability=('retweets', lambda s: s.astype(bool).mean()),
            mean_retweets=('retweets', 'mean'),
            median_retweets=('retweets', 'median'),
            max_retweets=('retweets', 'max'),
            # sources=('source', list),
        ).dropna().astype({'tweets': 'Int64', 'max_retweets': 'Int64'})
        stats = stats[stats.tweets >= min_size]  # Remove small classes
        return stats

    @classmethod
    def from_files(cls, graph_file, tweet_file):
        A, node_labels = read.labelled_graph(graph_file)
        tweets = read.tweets(tweet_file, node_labels)
        return cls(A, tweets)

    def sample_feature(self, size=None):
        """Return a sample of feature vectors."""
        return np.random.choice(self.features, size, p=self.params.freq)

    def sample_source(self, author_feature, size=None):
        """Sample uniformly from sources with author_feature."""
        return np.random.choice(self.sources[author_feature], size)

    def _default_sources(self, sources, feature, maybe_per_feature=False):
        if feature is None:
            return sources
        author_feature, _ = feature
        if sources is None:  # start once from each source with given author_feature
            return self.sources[author_feature]
        elif isinstance(sources, int):
            return self.sample_source(author_feature, size=sources)
        elif maybe_per_feature:
            try:
                try:
                    return sources[feature]
                except KeyError:
                    return sources[author_feature]
            except TypeError:
                pass
        return sources

    def _default_params(self, params, feature, maybe_per_feature=False):
        if feature is None:
            default_params = pd.Series({'edge_probability': pd.NA,
                                        'discount_factor': 1.,
                                        'max_retweets': 1000,
                                        'depth': 10,
                                        })
        else:
            default_params = self.params.loc[feature]
            if maybe_per_feature:
                try:
                    params = params[feature]
                except KeyError:
                    pass

        if not isinstance(params, pd.Series):
            params = pd.Series(params, index=default_params.index, dtype=object)
        return params.fillna(default_params)

    def edge_probability_from_retweet_probability(self, sources=None, eps=1e-5, features=None):
        """Find edge probability for given feature vector (or all if none given)."""
        if features is None:
            features = self.features
        return pd.Series((
            invert_monotone(lambda p: calculate_retweet_probability(self.A,
                                                                    self._default_sources(sources, f, True),
                                                                    p),
                            self.stats.loc[f, 'retweet_probability'],
                            0, 1,
                            eps) for f in features), index=features)

    @timecall
    def discount_factor_from_mean_retweets(self, sources=None, depth=10, samples=1000, eps=0.1, features=None):
        """Find discount factor for given feature vector (or all if none given)."""
        if features is None:
            features = self.features
        return pd.Series((
            invert_monotone(lambda d: self._sim(A=self.A,
                                                sources=self._default_sources(sources, f, True),
                                                p=self.params.loc[f, 'edge_probability'],
                                                discount=d,
                                                depth=depth,
                                                samples=samples)[0],
                            self.stats.loc[f, 'mean_retweets'],
                            0, 1,
                            eps=eps) for f in features), index=features)

    def simulate(self, feature=None, sources=None, params=None, samples=1, return_stats=True):
        """Simulate messages with given feature vector."""
        if feature:
            sources = self._default_sources(sources, feature)
        params = self._default_params(params, feature)

        return self._sim(self.A,
                         sources,
                         p=params.edge_probability,
                         discount=params.discount_factor,
                         depth=params.depth,
                         max_nodes=params.max_nodes,
                         samples=samples,
                         return_stats=return_stats)


if __name__ == "__main__":
    # pool = ray.util.multiprocessing.Pool(processes=32)
    # ray.init(num_cpus=50, memory=1000000000)
    # ray.init()
    datadir = '/Users/ian/Nextcloud'
    # datadir = '/home/sarming'
    # datadir = '/home/d3000/d300345'
    # read.adjlist(f'{datadir}/anonymized_outer_graph_neos_20200311.adjlist',
    #              save_as=f'{datadir}/outer_neos.npz')
    # A, node_labels = read.labelled_graph(f'{datadir}/outer_neos.npz')
    # tweets = read.tweets(f'{datadir}/authors_tweets_features_neos.csv', node_labels)
    # stats = Simulation.tweet_statistics(tweets)
    # features = stats.index
    sim = Simulation.from_files(f'{datadir}/outer_neos.npz', f'{datadir}/authors_tweets_features_neos.csv')
    # pool = Simulation.pool_from_files(f'{datadir}/outer_neos.npz', f'{datadir}/authors_tweets_features_neos.csv')
    # print(
    #     list(pool.map(lambda a, f: a.discount_factor_from_mean_retweets.remote(samples=1000, eps=0.1, features=[f]),
    #                   sim.features)))
    # print(sim.edge_probability_from_retweet_probability(sources=sim.sources))
    # print(sim.params.edge_probability)
    # ray.get(sim.discount_factor_from_mean_retweets(samples=1000, eps=0.1))
    # pool = multiprocessing.Pool(500, initializer=make_global, initargs=(sim.A,))
    sim.discount_factor_from_mean_retweets(samples=1000, eps=0.01, features=[('0010', '0010')])
    # sim.search_parameters(samples=1, eps=0.5,  feature=('0000', '0101') )
    # , feature=('0010', '1010'))
    # print(sim.features.loc[('0010', '1010')])
