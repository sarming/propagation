import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix, lil_matrix


@pytest.fixture
def graph_size():
    return 5


@pytest.fixture
def path(graph_size):
    m = lil_matrix((graph_size, graph_size))
    for i in range(graph_size - 1):
        m[i, [i + 1]] = 1.0
    return csr_matrix(m)


def tree(tree_degree, graph_size):
    tree_depth = int(math.ceil(math.log(graph_size, tree_degree)))
    m = lil_matrix((graph_size, graph_size))
    # t = {0: -1}
    first = 0  # index of first vertex at current level
    for depth in range(tree_depth):
        first_child = first + tree_degree ** depth  # index of first child
        for i in range(tree_degree ** (depth + 1)):  # iterate over all children
            child = first_child + i
            parent = first + i // tree_degree
            if child >= graph_size:
                break
            m[parent, child] = 1.0
            # t[child] = parent
        first = first_child
    # print(t)
    return csr_matrix(m)


@pytest.fixture
def binary_tree(graph_size):
    return tree(tree_degree=2, graph_size=graph_size)


@pytest.fixture
def empty_graph(graph_size):
    return csr_matrix((graph_size, graph_size))


@pytest.fixture
def zero_graph(graph_size):
    m = lil_matrix((graph_size, graph_size))
    for i in range(graph_size):
        for j in range(graph_size):
            m[i, j] = 0.0
    return csr_matrix(m)


@pytest.fixture
def clique(graph_size):
    m = lil_matrix((graph_size, graph_size))
    for i in range(graph_size):
        for j in range(graph_size):
            m[i, j] = 1.0
    return csr_matrix(m)


@pytest.fixture
def one_p_params():
    return {
        'edge_probability': 1.0,
        'discount_factor': 1.0,
        'corr': 0.0,
        'max_depth': 100,
        'max_nodes': 100,
        'at_least_one': True,
    }


@pytest.fixture
def seed_sequence():
    return np.random.SeedSequence(0)
