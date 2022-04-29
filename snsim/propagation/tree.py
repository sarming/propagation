from collections import defaultdict
from contextlib import contextmanager

import networkx as nx
import pandas as pd


def from_dict(d: dict):
    tree = nx.DiGraph()
    for v, parent in d.items():
        if parent == -1:
            tree.add_node(v)
            tree.graph["root"] = v
        else:
            tree.add_edge(parent, v)
    return tree


def depth_histogram(tree):
    lengths = nx.single_source_shortest_path_length(tree, tree.graph["root"])
    hist = defaultdict(int)
    for node, depth in lengths.items():
        hist[depth] += 1
    # hist[0] -= 1  # remove root
    return pd.Series(hist, dtype='Int64')


def shortest_path_histogram(graph, tree_or_tuple):
    hist = defaultdict(int)
    if isinstance(tree_or_tuple, tuple):  # Warning: we assume METIS ids here
        root = tree_or_tuple[0] - 1
        nodes = [x - 1 for x in tree_or_tuple[1]]
    else:
        root = tree_or_tuple.graph["root"]
        nodes = tree_or_tuple.nodes()
    for node in nodes:
        try:
            length = nx.shortest_path_length(graph, root, node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            print(f"No path from {root} to {node}")
            continue
        hist[length] += 1
    return pd.Series(hist, dtype='Int64')


def bfs_nodes(tree, node_labels=None):
    root = tree.graph["root"]
    edges = nx.bfs_edges(tree, root)
    if node_labels:
        return [node_labels[root]] + [node_labels[v] for u, v in edges]
    return [root] + [v for u, v in edges]


@contextmanager
def propagation_tree():
    from . import propagation

    orig = propagation.edge_propagate
    propagation.edge_propagate = propagation.edge_propagate_tree
    yield
    propagation.edge_propagate = orig


if __name__ == "__main__":
    from . import read, simulation

    datadir = 'data'
    graph, node_labels = read.metis(f'{datadir}/anon_graph_inner_neos_20201110.metis')
    tweets = read.tweets(f'{datadir}/sim_features_neos_20201110.csv', node_labels)
    sim = simulation.Simulation.from_tweets(graph, tweets)

    with propagation_tree():
        run = sim.run(10, samples=1)
        results = [
            (feature, from_dict(sample))
            for (feature, sources) in run
            for (source, samples) in sources
            for sample in samples
        ]

    for feature, tree in results:
        print(', '.join(map(str, list(feature) + bfs_nodes(tree, node_labels))))
        # print(nx.to_dict_of_dicts(tree))

    features, trees = list(zip(*results))
    hist = pd.concat(map(depth_histogram, trees), axis=1).fillna(0).transpose()
    print(hist)
