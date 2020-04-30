import re
import networkx as nx
import pandas as pd
import numpy as np
import scipy as sp


def adjlist(file, save_as=None):
    graph = nx.read_adjlist(file, nodetype=int, create_using=nx.DiGraph)
    if save_as:
        save_labelled_graph(save_as, nx.to_scipy_sparse_matrix(graph), graph.nodes())
    return graph


def save_labelled_graph(filename, A, node_labels):
    np.savez_compressed(filename, data=A.data, indices=A.indices, indptr=A.indptr, shape=A.shape, node_list=node_labels)
    # np.savez(filename, data=A.data, indices=A.indices, indptr=A.indptr, shape=A.shape, node_labels=node_labels)


def labelled_graph(filename):
    loader = np.load(filename)
    A = sp.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                             shape=loader['shape'])
    node_list = loader['node_labels']
    return A, node_list


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

