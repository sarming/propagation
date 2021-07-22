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


def bfs_nodes(tree, root):
    edges = nx.bfs_edges(tree, root)
    return [root] + [v for u, v in edges]


if __name__ == "__main__":
    from itertools import chain, starmap
    from simulation import Simulation
    import propagation

    propagation.edge_propagate = propagation.edge_propagate_tree
    datadir = 'data'
    graph = f'{datadir}/anonymized_inner_graph_vegan_20200407.npz'
    tweets = f'{datadir}/sim_features_vegan_20200407.csv'
    sim = Simulation.from_files(graph, tweets)
    results = chain(*chain(*sim.run(10, samples=1)))  # Flatten
    trees = list(map(from_dict, results))
    for tree, root in trees:
        print(', '.join(map(str, bfs_nodes(tree, root))))
    hist = pd.concat(starmap(depth_histogram, trees), axis=1).fillna(0).transpose()
    print(hist)
