import pytest

import snsim.propagation.propagation as p


class TestEdgePropagate:
    def test_one_path(self, path):
        assert p.edge_propagate(path, source=0, p=1.0) == 4
        assert p.edge_propagate(path, source=2, p=1.0) == 2
        assert p.edge_propagate(path, source=4, p=1.0) == 0

    def test_one_tree(self, binary_tree):
        assert p.edge_propagate(binary_tree, source=0, p=1.0) == 4

    def test_one_clique(self, clique):
        assert p.edge_propagate(clique, source=0, p=1.0) == 4
        assert p.edge_propagate(clique, source=4, p=1.0) == 4

    def test_zero_path(self, path):
        assert p.edge_propagate(path, source=0, p=0.0) == 0

    def test_zero_tree(self, binary_tree):
        assert p.edge_propagate(binary_tree, source=0, p=0.0) == 0

    def test_zero_clique(self, clique):
        assert p.edge_propagate(clique, source=0, p=0.0) == 0

    def test_zero_graph(self, zero_graph):
        assert p.edge_propagate(zero_graph, source=0, p=1.0) == 0


class TestEdgePropagateTree:
    def test_one_path(self, path):
        assert p.edge_propagate_tree(path, source=0, p=1.0) == {0: -1, 1: 0, 2: 1, 3: 2, 4: 3}
        assert p.edge_propagate_tree(path, source=2, p=1.0) == {2: -1, 3: 2, 4: 3}

    def test_zero_path(self, path):
        assert p.edge_propagate_tree(path, source=0, p=0.0) == {0: -1}

    def test_one_tree(self, binary_tree):
        assert p.edge_propagate_tree(binary_tree, source=0, p=1.0) == {
            0: -1,
            1: 0,
            2: 0,
            3: 1,
            4: 1,
        }
        assert p.edge_propagate_tree(binary_tree, source=1, p=1.0) == {1: -1, 3: 1, 4: 1}

    def test_zero_path(self, path):
        assert p.edge_propagate_tree(path, source=0, p=0.0) == {0: -1}


class TestSimulateStats:
    @pytest.fixture
    def sim_one(self, one_p_params, seed_sequence):
        return lambda graph, sources, samples: p.simulate(
            graph, one_p_params, sources, samples, True, seed_sequence
        )

    def test_one_path(self, path, sim_one):
        assert sim_one(path, sources=[0, 0], samples=3) == (4.0, 1.0)
        assert sim_one(path, sources=[2, 2, 2], samples=10) == (2.0, 1.0)
        assert sim_one(path, sources=[4, 4], samples=27) == (0.0, 0.0)
        assert sim_one(path, sources=[0, 4], samples=5) == (2.0, 0.5)
        assert sim_one(path, sources=[0, 1, 2], samples=5) == (3.0, 1.0)
        assert sim_one(path, sources=[0, 1, 2, 3, 4], samples=5) == (2.0, 0.8)
