import pytest
from scipy.sparse import csr_matrix, lil_matrix


@pytest.fixture(scope="session")
def path(path_length=5):
    m = lil_matrix((path_length, path_length))
    for i in range(path_length - 1):
        m[i, [i + 1]] = 1.0
    return csr_matrix(m)
