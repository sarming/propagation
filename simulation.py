from functools import wraps

import numpy as np
import pandas as pd
# from profilehooks import timecall

import propagation
import read


def calculate_retweet_probability(A, sources, p, at_least_one):
    """Return average number of retweeted messages when starting from sources using edge probability p.

    Args:
        A: Adjacency matrix of graph.
        sources: List of source nodes, one per tweet.
        p: Edge probability.

    Returns:
        mean_{x in sources} 1-(1-p)^{deg-(x)}
        This is the expected value of simulate(A, sources, p, depth=1)[1].
    """
    if at_least_one:
        return p
    return sum(1 - (1 - p) ** float(A.indptr[x + 1] - A.indptr[x]) for x in sources) / len(sources)


def invert_monotone(fun, goal, lb, ub, eps, logging=False):
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
        logging: Print function calls and results.

    Returns: x s.t. lb<x<ub and there is y with |x-y|<=eps and fun(y)=goal
    """
    mid = (ub + lb) / 2
    if ub - lb < eps:
        return mid
    if logging:
        print(f'f({mid})=', end='')
    f = fun(mid)
    if logging:
        print(f"{f}{'<' if f < goal else '>'}{goal} [{lb},{ub}]")
    if f < goal:
        return invert_monotone(fun, goal, mid, ub, eps, logging)
    else:
        return invert_monotone(fun, goal, lb, mid, eps, logging)


def logging(fun):
    """Simple function decorator that prints every function invocation."""

    @wraps(fun)
    def wrapper(*args, **kwargs):
        print(f'calling {fun.__name__}({args},{kwargs})')
        return fun(*args, **kwargs)

    return wrapper


def tweet_statistics(tweets, min_size=10):
    """Return statistics dataframe for tweets. Throw away feature classes smaller than min_size."""
    stats = tweets.groupby(['author_feature', 'tweet_feature']).agg(
        tweets=('retweets', 'size'),
        retweet_probability=('retweets', lambda s: s.astype(bool).mean()),
        mean_retweets=('retweets', 'mean'),
        median_retweets=('retweets', 'median'),
        max_retweets=('retweets', 'max'),
        # sources=('source', list),
    ).dropna().astype({'tweets': 'Int64', 'max_retweets': 'Int64'})
    stats = stats[stats.tweets >= min_size]  # Remove small classes
    # stats.to_csv('data/tmp_stats.csv')
    return stats


def tweet_sources(tweets):
    """Return map from author feature to list of sources with this feature."""
    return tweets.dropna().groupby('author_feature')['source'].unique()


class Simulation:
    def __init__(self, A, stats, sources, params=None, simulator=propagation.simulate, seed=None):
        self.A = A
        self.stats = stats
        self.sources = sources
        self.params = pd.DataFrame({'freq': self.stats.tweets / self.stats.tweets.sum(),
                                    'edge_probability': np.NaN,  # will be calculated below
                                    'at_least_one': True,
                                    'discount_factor': 1.0,
                                    'corr': 0.0,
                                    'max_nodes': 10 * self.stats.max_retweets,
                                    'max_depth': 50,
                                    })
        self.features = self.stats.index

        self.simulator = simulator
        # self.simulator = parallel.ray_simulator()
        # self.simulator = parallel.pool_simulator(self.A)
        # self.simulator = logging(self.simulator)

        self.params['edge_probability'] = self.edge_probability_from_retweet_probability()
        if params is not None:
            self.params.update(params)

        if not isinstance(seed, np.random.SeedSequence):
            seed = np.random.SeedSequence(seed)  # Random if seed is None
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    @classmethod
    def from_files(cls, graph_file, tweet_file, simulator=propagation.simulate, seed=None):
        """Return Simulation using for the given files."""
        A, node_labels = read.labelled_graph(graph_file)
        tweets = read.tweets(tweet_file, node_labels)
        return cls.from_tweets(A, tweets, simulator, seed)

    @classmethod
    def from_tweets(cls, A, tweets, simulator=propagation.simulate, seed=None):
        """Return Simulation using for the given files."""
        stats = tweet_statistics(tweets)
        sources = tweet_sources(tweets)
        return cls(A, stats, sources, None, simulator, seed)

    def sample_feature(self, size=None):
        """Return a sample of feature vectors (according to feature distribution)."""
        return self.rng.choice(self.features, size, p=self.params.freq)

    def sample_source(self, author_feature, size=None):
        """Sample uniformly from sources with author_feature."""
        return self.rng.choice(self.sources[author_feature], size)

    def _default_sources(self, sources, feature):
        if feature is None:
            return sources
        author_feature, _ = feature
        if sources is None:  # start once from each source with given author_feature
            return self.sources[author_feature]
        elif isinstance(sources, int):
            return self.sample_source(author_feature, size=sources)
        try:
            try:
                return sources[feature]
            except (IndexError, KeyError, TypeError):
                return sources[author_feature]
        except (IndexError, KeyError, TypeError):
            pass
        return sources

    def _default_params(self, params, feature):
        if feature is None:
            default_params = pd.Series({'edge_probability': pd.NA,
                                        'at_least_one': True,
                                        'discount_factor': 1.0,
                                        'corr': 0.0,
                                        'max_nodes': 1000,
                                        'max_depth': 50,
                                        }, dtype=object)
        else:
            default_params = self.params.astype('object').loc[feature]
            try:
                params = params.loc[feature]
            except (AttributeError, IndexError, KeyError, TypeError):
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
                                                                    self._default_sources(sources, f),
                                                                    p,
                                                                    self.params.at[f, 'at_least_one']),
                            self.stats.loc[f, 'retweet_probability'],
                            0, 1,
                            eps) for f in features), index=features)

    # @timecall
    def learn(self, param, goal_stat, lb, ub, eps, sources, samples, features, params):
        def set_param(value, feature):
            p = self._default_params(params, feature)
            p[param] = value
            return p

        def goal(results):
            return list(results)[0 if goal_stat == 'mean_retweets' else 1]

        if features is None:
            features = self.features
        return pd.Series((
            invert_monotone(lambda x: goal(self.simulator(A=self.A,
                                                          sources=self._default_sources(sources, f),
                                                          params=set_param(x, f),
                                                          samples=samples,
                                                          seed=self.seed.spawn(1)[0])),
                            self.stats.loc[f, goal_stat],
                            lb, ub,
                            eps=eps, logging=True) for f in features), index=features)

    def discount_factor_from_mean_retweets(self, sources=None, params=None, samples=1000, eps=0.1, features=None):
        """Find discount factor for given feature vector (or all if none given)."""
        return self.learn('discount_factor', 'mean_retweets', 0., 1., sources=sources, params=params, samples=samples,
                          eps=eps, features=features)

    def corr_from_mean_retweets(self, sources=None, params=None, samples=1000, eps=0.1, features=None):
        """Find corr for given feature vector (or all if none given)."""
        return self.learn('corr', 'mean_retweets', 0., .01, sources=sources, params=params, samples=samples, eps=eps,
                          features=features)

    # @timecall
    def simulate(self, feature=None, sources=None, params=None, samples=1, return_stats=True):
        """Simulate message with given feature vector.

        Args:
            feature (pair): author_feature and tweet_feature.
            sources: list of sources or number of sources (may also be dict from (author) feature to sources)
            params (dict-like): overwrite parameters, missing values will be filled by params for feature (may also be dict from feature to parameters)
            samples: number of samples
            return_stats: return (mean_retweets, retweet_probability) instead

        Returns:
            List of lists of retweet counts.
        """
        sources = self._default_sources(sources, feature)
        params = self._default_params(params, feature)
        # print(params)

        return self.simulator(self.A,
                              sources,
                              params,
                              samples=samples,
                              return_stats=return_stats,
                              seed=self.seed.spawn(1)[0])

    def run(self, num_features, num_sources=1, params=None, samples=100):
        for feature in self.sample_feature(num_features):
            sources = self._default_sources(num_sources, feature)
            yield feature, zip(sources, self.simulate(feature, sources=sources, params=params, samples=samples,
                                                      return_stats=False))


if __name__ == "__main__":
    # import sys, ray
    # ray.init(address=sys.argv[1], redis_password=sys.argv[2])
    # ray.init(num_cpus=50, memory=1000000000)
    datadir = 'data'
    # datadir = '/Users/ian/Nextcloud'
    # datadir = '/home/sarming'
    # datadir = '/home/d3000/d300345'
    # read.adjlist(f'{datadir}/anonymized_outer_graph_neos_20200311.adjlist',
    #              save_as=f'{datadir}/outer_neos.npz')
    # A, node_labels = read.labelled_graph(f'{datadir}/outer_neos.npz')
    # tweets = read.tweets(f'{datadir}/sim_features_neos.csv', node_labels)
    # stats = Simulation.tweet_statistics(tweets)
    # features = stats.index
    sim = Simulation.from_files(f'{datadir}/outer_neos.npz', f'{datadir}/sim_features_neos_20200311.csv')


    # pool = Simulation.pool_from_files(f'{datadir}/outer_neos.npz', f'{datadir}/sim_features_neos.csv')
    # print(
    #     list(pool.map(lambda a, f: a.discount_factor_from_mean_retweets.remote(samples=1000, eps=0.1, features=[f]),
    #                   sim.features)))
    # print(sim.edge_probability_from_retweet_probability(sources=sim.sources))
    # print(sim.params.edge_probability)
    # ray.get(sim.discount_factor_from_mean_retweets(samples=1000, eps=0.1))
    # pool = multiprocessing.Pool(500, initializer=make_global, initargs=(sim.A,))
    # sim.simulator = parallel.ray_simulator()

    # sim.edge_probability_from_retweet_probability(features=[('0000', '0001')], sources=sim.sources['0101'])
    # sim.corr_from_mean_retweets(samples=1000, eps=0.001, features=[('0010', '0010')])
    # sim.search_parameters(samples=1, eps=0.5,  feature=('0000', '0101') )
    # , feature=('0010', '1010'))
    # print(sim.features.loc[('0010', '1010')])
    def most_frequent():
        s = sorted(sim.features, key=lambda f: sim.stats.loc[f].tweets)
        return list(s)


    # @timecall
    def run():
        for feature in most_frequent()[:10]:
            stats = sim.stats.loc[feature]
            result = sim.simulate(feature, sources=1000, samples=1000)
            print(f'{feature}: {stats.mean_retweets} vs {result[0]}, {stats.retweet_probability} vs {result[1]}')


    run()
