import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp


def adjlist(file, save_as=None):
    graph = nx.read_adjlist(file, nodetype=int, create_using=nx.DiGraph)
    A = nx.to_scipy_sparse_matrix(graph)
    node_labels = graph.nodes()
    if save_as:
        save_labelled_graph(save_as, A, node_labels)
    return A, node_labels


def save_labelled_graph(filename, A, node_labels, compressed=True):
    savez = np.savez_compressed if compressed else np.savez
    savez(filename, data=A.data, indices=A.indices, indptr=A.indptr, shape=A.shape, node_labels=node_labels)


def labelled_graph(filename):
    loader = np.load(filename)
    A = sp.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                             shape=loader['shape'])
    node_labels = loader['node_labels']
    return A, node_labels


def metis(filename, zero_based=False):
    with open(filename) as f:
        (n, m) = f.readline().split()
        n = int(n)
        mtx = sp.sparse.lil_matrix((n, n))
        for (node, neighbors) in enumerate(f.readlines()):
            neighbors = [int(v) - (1 if not zero_based else 0) for v in neighbors.split()]
            mtx[node, neighbors] = 1.
        node_labels = range(n) if zero_based else range(1, n+1)
        return mtx.tocsr(), node_labels


def tweets(file, node_labels):
    def str_cat_series(*series):
        # return list(map(str,zip(*series))) # to support nonbinary features
        series = list(map(lambda x: x.apply(str), series))
        return series[0].str.cat(series[1:]).astype("category")

    csv = pd.read_csv(file)

    csv['author_feature'] = str_cat_series(csv['verified'], csv['activity'], csv['defaultprofile'], csv['userurl'])
    csv['tweet_feature'] = str_cat_series(csv['hashtag'], csv['tweeturl'], csv['mentions'], csv['media'])

    reverse = {node: idx for idx, node in enumerate(node_labels)}
    csv['source'] = pd.Series((reverse.get(author, None) for author in csv['author']), dtype='Int64')

    return csv[['source', 'author_feature', 'tweet_feature', 'retweets']]


def stats(file):
    # TODO optionally remove max_retweets
    stats = pd.read_csv(file, dtype={'author_feature': str,
                                     'tweet_feature': str,
                                     'retweets': 'Int64',
                                     'tweets': 'Int64',
                                     'max_retweets': 'Int64'})
    stats.set_index(['author_feature', 'tweet_feature'], inplace=True)
    return stats


def source_map(file):
    sources = pd.read_csv(file, dtype={'author_feature': str, 'source': 'Int64'})
    return sources.dropna().groupby('author_feature')['source'].unique()


def save_source_map(file, source_map):
    source_map.explode().reset_index(level=0).set_index('source').to_csv(file)


def single_param(file):
    discount = pd.read_csv(file, dtype={'author_feature': str, 'tweet_feature': str})
    discount.set_index(['author_feature', 'tweet_feature'], inplace=True)
    return discount.squeeze()


if __name__ == "__main__":
    for g in ['fpoe_20200311', 'neos_20200311', 'bvb_20200409', 'schalke_20200409', 'vegan_20200407']:
        print(g)
        adjlist(f'data/anonymized_outer_graph_{g}.adjlist', save_as=f'outer_{g}.npz')
