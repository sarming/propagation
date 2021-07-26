import time

import numpy as np
import scipy.sparse as sp
from mpi4py import MPI

import mpi
import propagation
import read
from simulation import Simulation


def top_k_users(graph, k):
    _, diag = sp.csgraph.laplacian(graph.transpose(), return_diag=True)
    diag = -diag
    return diag.argsort()[:k]


if MPI.COMM_WORLD.Get_rank() == 0:
    graph, node_labels = read.adjlist(f'data/anonymized_inner_graph_neos_20200311.adjlist')
    tweets = read.tweets(f'data/sim_features_neos_20200311.csv', node_labels)
    sim = Simulation.from_tweets(graph, tweets, seed=None)  # Put seed here

    # Play around with these parameters
    sources = np.repeat(top_k_users(graph, 30), 10)  # Repeat top 30 vertices 10 times
    params = {'edge_probability': 0.6,  # Will influence runtime edge_propagate
              'max_nodes': graph.shape[0]  # Disable early cutoff by setting to maximum
              }
    samples = 100  # Will influence task size
else:
    sim = None
    propagation.compile()

with mpi.futures(sim, chunksize=1) as sim:
    if sim is not None:
        t = time.time()
        r = list(sim.simulate(sources=sources, params=params, samples=samples))
        print(time.time() - t)
        print(r)
