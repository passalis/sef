from sef_dr.similarity import fast_distance_matrix, fast_heat_similarity_matrix
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


def test_distance_calculations():
    """
    Tests the implementation of fast distance calculations with the PyTorch
    :return:
    """
    np.random.seed(1)

    # Create random data vectors
    A = np.random.randn(10, 23)
    B = np.random.randn(5, 23)

    sef_dists = fast_distance_matrix(A, B)

    assert sef_dists.shape[0] == 10
    assert sef_dists.shape[1] == 5

    dists = pairwise_distances(A, B)

    assert np.sum((sef_dists-dists)*2) < 1e-3


def test_similarity_calculations():
    """
    Tests the implementation of fast similarity calculations with the PyTorch
    :return:
    """
    np.random.seed(1)

    # Create random data vectors
    for sigma in [0.01, 0.1, 0.5, 1]:
        A = np.random.randn(10, 23)
        sef_sim = fast_heat_similarity_matrix(A, sigma)

        assert sef_sim.shape[0] == 10
        assert sef_sim.shape[1] == 10

        sim = np.exp(-pairwise_distances(A, A)**2/sigma**2)
        assert np.sum((sef_sim-sim)*2) < 1e-3

