import snsim.propagation.propagation as p


def test_one_path(path):
    assert p.edge_propagate(path, source=0, p=1.0) == 4


def test_zero_path(path):
    assert p.edge_propagate(path, source=0, p=0.0) == 0
