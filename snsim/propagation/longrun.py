#!/usr/bin/env python
import time

import numpy as np
import scipy.sparse as sp
from mpi4py import MPI

# https://stackoverflow.com/a/28154841/153408
if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    __package__ = "snsim.propagation"
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.path.pardir))

from . import mpi, propagation, read
from .simulation import Simulation


def top_k_users(graph, k):
    _, diag = sp.csgraph.laplacian(graph.transpose(), return_diag=True)
    diag = -diag
    return diag.argsort()[:k]


if MPI.COMM_WORLD.Get_rank() == 0:
    graph, node_labels = read.metis('data/anon_graph_inner_neos_20201110.metis')
    tweets = read.tweets('data/sim_features_neos_20201110.csv', node_labels)
    sim = Simulation.from_tweets(graph, tweets, seed=None)  # Put seed here

    # Play around with these parameters
    sources = np.repeat(top_k_users(graph, 30), 10)  # Repeat top 30 vertices 10 times
    params = {
        'edge_probability': 0.999999,  # Will influence runtime edge_propagate
        'max_nodes': graph.shape[0],  # Disable early cutoff by setting to maximum
        'max_depth': graph.shape[0],
        # 'at_least_one': False,
    }
    samples = 100  # Will influence task size
else:
    sim = None
    propagation.compile()

with mpi.futures(sim, chunksize=1) as sim:
    if sim is not None:
        t = time.time()
        # r = list(list(x) for x in sim.simulate(params=params, sources=sources, samples=samples, return_stats=False))
        r = list(sim.simulate(params=params, sources=sources, samples=samples))
        print(time.time() - t)
        print(r)
