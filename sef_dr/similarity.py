# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt

import theano
import theano.tensor as T
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def mean_data_distance(data):
    """
    Calculates the mean distance between a set of data points
    :param data: 
    :return: 
    """
    mean_distance = np.mean(pairwise_distances(data))
    return mean_distance


def sym_distance_matrix(A, B):
    """
    Defines the symbolic matrix that contains the distances between the vectors of A and B
    :param A:
    :param B:
    :return:
    """
    aa = T.sum(A * A, axis=1)
    bb = T.sum(B * B, axis=1)
    AB = T.dot(A, T.transpose(B))

    AA = T.transpose(T.tile(aa, (bb.shape[0], 1)))
    BB = T.tile(bb, (aa.shape[0], 1))

    D = AA + BB - 2 * AB
    D = T.fill_diagonal(D, 0)
    D = T.sqrt(T.maximum(D, 0))

    return D


def sym_similarity_matrix(X, sigma):
    """
    Defines the self similarity matrix using the heat kernel
    :param X:
    :param sigma:
    :return:
    """
    D = sym_distance_matrix(X, X)
    return T.exp(-D ** 2 / (sigma ** 2))


# Dirty trick
# Compile a fast similarity matrix search function
X = T.matrix('X', dtype=theano.config.floatX)
sigma = T.scalar('g', dtype=theano.config.floatX)
fast_similarity_matrix_fn = theano.function([X, sigma], sym_similarity_matrix(X, sigma))
fast_distance_matrix_fn = theano.function([X], sym_distance_matrix(X, X))
