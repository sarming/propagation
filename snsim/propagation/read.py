import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix


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
            mtx[nodes[0], sorted(nodes[1:])] = 1.0
    mtx = mtx.tocsr()

    save_labelled_graph(save_as, mtx, node_labels)
    return mtx, node_labels


def save_labelled_graph(filename, A, node_labels, compressed=False):
    if filename:
        savez = np.savez_compressed if compressed else np.savez
        if isinstance(node_labels, range):  # Store just starting index for ranges
            node_labels = node_labels.start
        savez(
            filename,
            data=False,
            indices=A.indices,
            indptr=A.indptr,
            shape=A.shape,
            node_labels=node_labels,
        )


def labelled_graph(filename):
    loader = np.load(filename)
    data = loader['data']
    if not data.shape:
        data = np.full(loader['indices'].shape, 1.0)
    A = csr_matrix((data, loader['indices'], loader['indptr']), shape=loader['shape'])

    node_labels = loader['node_labels']
    if not node_labels.shape and loader['shape'][0]:  # Restore saved range
        node_labels = range(node_labels, loader['shape'][0] + node_labels)
    return A, node_labels


def metis(filename, zero_based=False, save_as=None):
    with open(filename) as f:
        (n, m) = f.readline().split()
        n = int(n)
        node_labels = range(n) if zero_based else range(1, n + 1)

        mtx = lil_matrix((n, n))
        for (node, neighbors) in enumerate(f.readlines()):
            neighbors = [int(v) - (1 if not zero_based else 0) for v in neighbors.split()]
            if neighbors:
                mtx[node, sorted(neighbors)] = 1.0
        mtx = mtx.tocsr()

        save_labelled_graph(save_as, mtx, node_labels)
        return mtx, node_labels


def str_cat_series(*series):
    # return list(map(str,zip(*series))) # to support nonbinary features
    series = list(map(lambda x: x.apply(str), series))
    return series[0].str.cat(series[1:]).astype('category')


def str_cat_df(df, cols):
    series = (df[c] for c in cols)
    return str_cat_series(*series)


def tweets(file, node_labels, id_type='metis'):
    csv = pd.read_csv(file)

    csv['author_feature'] = str_cat_df(csv, ['verified', 'activity', 'defaultprofile', 'userurl'])
    csv['tweet_feature'] = str_cat_df(csv, ['hashtag', 'tweeturl', 'mentions', 'media'])

    if isinstance(node_labels, range):
        a, b = node_labels.start, node_labels.stop
        reverse = lambda x: x - a if a <= x < b else None
    else:
        r = {node: idx for idx, node in enumerate(node_labels)}
        reverse = lambda x: r.get(x, None)

    csv['source'] = pd.Series(
        (reverse(author) for author in csv[f'author_{id_type}']), dtype='Int64'
    )

    return csv[['source', 'author_feature', 'tweet_feature', 'retweets']]


def stats(file):
    stats = pd.read_csv(
        file,
        dtype={
            'author_feature': str,
            'tweet_feature': str,
            'tweets': 'Int64',
            'retweet_probability': float,
            'mean_retweets': float,
            'max_retweets': 'Int64',
        },
    )
    # set default max retweets if not provided
    if 'max_retweets' not in stats.columns:
        stats['max_retweets'] = 100
    stats.set_index(['author_feature', 'tweet_feature'], inplace=True)
    return stats


def source_map(file):
    sources = pd.read_csv(file, dtype={'author_feature': str, 'source': 'Int64'})
    return sources.dropna().groupby('author_feature')['source'].unique()


def save_source_map(file, source_map: pd.DataFrame):
    import warnings

    with warnings.catch_warnings():  # Upstream bug: https://github.com/pandas-dev/pandas/issues/45629
        warnings.simplefilter("ignore", category=FutureWarning)
        source_map.explode().reset_index(level=0).set_index('source').to_csv(file)


def single_param(file):
    discount = pd.read_csv(file, dtype={'author_feature': str, 'tweet_feature': str})
    discount.set_index(['author_feature', 'tweet_feature'], inplace=True)
    return discount.squeeze()


def params(file):
    params = pd.read_csv(file, dtype={'author_feature': str, 'tweet_feature': str})
    params.set_index(['author_feature', 'tweet_feature'], inplace=True)
    params.freq = params.freq / params.freq.sum()  # renormalize rel. frequencies
    return params


if __name__ == '__main__':
    from glob import glob

    for f in glob('data/*.metis'):
        print(f)
        metis(f, save_as=f.replace('metis', 'npz'))
