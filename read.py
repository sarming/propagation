import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix


def adjlist(filename, save_as=None):
    from collections import OrderedDict

    labels = OrderedDict()
    with open(filename) as f:
        for line in f:
            for i in line.split():
                labels[int(i)] = None
    for i, label in enumerate(labels.keys()):
        labels[label] = i
    node_labels = list(labels.keys())

    n = len(node_labels)
    mtx = lil_matrix((n, n))
    with open(filename) as f:
        for line in f:
            nodes = [labels[int(v)] for v in line.split()]
            mtx[nodes[0], sorted(nodes[1:])] = 1.
    mtx = mtx.tocsr()

    # graph = nx.read_adjlist(filename, nodetype=int, create_using=nx.DiGraph)
    # A = nx.to_scipy_sparse_matrix(graph)
    # assert node_labels == list(graph.nodes())
    # assert (mtx != A).nnz == 0

    if save_as:
        save_labelled_graph(save_as, mtx, node_labels)
    return mtx, node_labels


def save_labelled_graph(filename, A, node_labels, compressed=True):
    savez = np.savez_compressed if compressed else np.savez
    savez(filename, data=A.data, indices=A.indices, indptr=A.indptr, shape=A.shape, node_labels=node_labels)


def labelled_graph(filename):
    loader = np.load(filename)
    A = csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                   shape=loader['shape'])
    node_labels = loader['node_labels']
    return A, node_labels


def metis(filename, zero_based=False, save_as=None):
    with open(filename) as f:
        (n, m) = f.readline().split()
        n = int(n)
        mtx = lil_matrix((n, n))
        for (node, neighbors) in enumerate(f.readlines()):
            neighbors = [int(v) - (1 if not zero_based else 0) for v in neighbors.split()]
            if neighbors:
                mtx[node, sorted(neighbors)] = 1.
        node_labels = range(n) if zero_based else range(1, n + 1)
        mtx = mtx.tocsr()
        if save_as:
            save_labelled_graph(save_as, mtx, node_labels)
        return mtx, node_labels


def tweets(file, node_labels, id_type='metis'):
    def str_cat_series(*series):
        # return list(map(str,zip(*series))) # to support nonbinary features
        series = list(map(lambda x: x.apply(str), series))
        return series[0].str.cat(series[1:]).astype("category")

    csv = pd.read_csv(file)

    csv['author_feature'] = str_cat_series(csv['verified'], csv['activity'], csv['defaultprofile'], csv['userurl'])
    csv['tweet_feature'] = str_cat_series(csv['hashtag'], csv['tweeturl'], csv['mentions'], csv['media'])

    reverse = {node: idx for idx, node in enumerate(node_labels)}
    csv['source'] = pd.Series((reverse.get(author, None) for author in csv[f'author_{id_type}']), dtype='Int64')

    return csv[['source', 'author_feature', 'tweet_feature', 'retweets']]


def stats(file):
    stats = pd.read_csv(file, dtype={'author_feature': str,
                                     'tweet_feature': str,
                                     'tweets': 'Int64',
                                     'retweet_probability': float,
                                     'mean_retweets': float,
                                     'max_retweets': 'Int64'})
    # set default max retweets if not provided
    if 'max_retweets' not in stats.columns:
        stats['max_retweets'] = 100
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
    import networkx as nx
    import scipy.sparse.linalg
    import ast

    a, labels_a = adjlist('data/anonymized_inner_graph_neos_20200311.adjlist')
    b, labels_b = metis(f'data/anon_graph_inner_neos_20201110.metis')
    a = nx.from_scipy_sparse_matrix(a, create_using=nx.DiGraph)
    b = nx.from_scipy_sparse_matrix(b, create_using=nx.DiGraph)
    conv = pd.read_csv('data/sim_features_neos_20210729.csv', converters={'retweeter_adjlist': ast.literal_eval,
                                                                          'retweeter_metis': ast.literal_eval})
    r = l = t = 0
    for _, c in conv.iterrows():
        if not int(c['author_adjlist']) in labels_a:
            continue
        x = labels_a.index(c['author_adjlist'])
        y = labels_b.index(c['author_metis'])
        ra = c['retweeter_adjlist']
        assert a.out_degree(x) == b.out_degree(y)

        if not (ra and ra[0] in labels_a):
            continue
        t += 1
        ra = labels_a.index(ra[0])
        rb = labels_b.index(c['retweeter_metis'][0])
        print(f'{x} {a.out_degree(x)} {a.has_edge(x, ra)}, {y} {b.out_degree(y)} {b.has_edge(y, rb)} {rb}')
        assert a.has_edge(x, ra) == b.has_edge(y, rb)
        if a.has_edge(x, ra):
            r += 1
        if a.has_edge(ra, x):
            l += 1
    print(f'{t} {r} {l}')
    print(nx.is_isomorphic(a, b))
