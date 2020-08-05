import numpy as np
import scipy.sparse
import scipy as sp
import pandas as pd
from propagation import edge_sample


def live_propagate(A, tweet_stream):
    """Time stepped version of Twitter simulation.

    Args:
        A: sparse adjacency matrix
        tweet_stream: stream of lists of (source, params) tuples

    Yields:
        list of (tweet_id, from, to) tuples

    """
    tweet_id = 0
    tweets = []
    for new_tweets in tweet_stream:
        updates = []

        for tweet in tweets:
            params = tweet['params']
            tweet['age'] += 1
            if tweet['age'] > params.max_depth:
                tweet['leaves'] = set()

            next_leaves = set()
            for node in tweet['leaves']:
                children = set(edge_sample(A, node, p=params.edge_probability, at_least_one=False))
                children -= tweet['visited']
                next_leaves |= children
                tweet['visited'] |= children
                for c in children:
                    updates.append((tweet['id'], node, c))
                if len(tweet['visited']) > params.max_nodes:
                    next_leaves = set()
                    break
            tweet['leaves'] = next_leaves
            tweet['params'].edge_probability *= params.discount_factor

        for (source, params) in new_tweets:
            tweet = {'id': tweet_id,
                     'params': params,
                     'leaves': {source},
                     'visited': {source},
                     'age': 0,
                     }
            tweets.append(tweet)
            updates.append((tweet_id, None, source))
            tweet_id += 1

        tweets = list(filter(lambda t: len(t['leaves']) > 0, tweets))
        yield updates


def uniform_tweet_stream(num_sources, params, skip=0):
    """ Yields tweets with random source and fixed params."""
    while True:
        yield [(np.random.randint(num_sources), params)]
        for _ in range(skip):
            yield []


if __name__ == "__main__":
    n = 50

    A = np.random.randint(2, size=(n, n))
    A = sp.sparse.csr_matrix(A)
    # A = nx.to_scipy_sparse_matrix(graph)

    params = pd.Series({'edge_probability': 0.05,
                        'discount_factor': 1.,
                        'max_nodes': n,
                        'max_depth': 100,
                        }, dtype=object)

    tweet_stream = uniform_tweet_stream(n, params, 2) #

    for updates in live_propagate(A, tweet_stream):
        print(updates)

