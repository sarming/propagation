import networkx as nx
import pandas as pd
from collections import defaultdict


def from_dict(d: dict):
    tree = nx.DiGraph()
    root = None
    for v, parent in d.items():
        if parent == -1:
            tree.add_node(v)
            root = v
        else:
            tree.add_edge(parent, v)
    return tree, root


def depth_histogram(tree, root):
    lengths = nx.single_source_shortest_path_length(tree, root)
    hist = defaultdict(int)
    for node, depth in lengths.items():
        hist[depth] += 1
    # return hist
    return pd.Series(hist, dtype='Int64')


def bfs_nodes(tree, root, node_labels=None):
    edges = nx.bfs_edges(tree, root)
    if node_labels:
        return [node_labels[root]] + [node_labels[v] for u, v in edges]
    return [root] + [v for u, v in edges]


if __name__ == "__main__":
    from itertools import starmap
    from simulation import Simulation
    import propagation
    import read

    propagation.edge_propagate = propagation.edge_propagate_tree

    datadir = 'data'
    graph, node_labels = read.adjlist(f'{datadir}/anonymized_inner_graph_vegan_20200407.adjlist')
    tweets = read.tweets(f'{datadir}/sim_features_vegan_20200407.csv', node_labels)
    sim = Simulation.from_tweets(graph, tweets)

    run = sim.run(10, samples=1)
    results = [(feature, from_dict(sample)) for feature, sources in run for source in sources for sample in source]
    for feature, (tree, root) in results:
        print(', '.join(map(str, list(feature) + bfs_nodes(tree, root, node_labels))))
        # print(nx.to_dict_of_dicts(tree))

    trees = list(zip(*results))[1]
    hist = pd.concat(starmap(depth_histogram, trees), axis=1).fillna(0).transpose()
    print(hist)
