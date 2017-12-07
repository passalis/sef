# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics.pairwise import pairwise_distances


def mean_data_distance(data):
    """
    Calculates the mean distance between a set of data points
    :param data:
    :return:
    """
    mean_distance = np.mean(pairwise_distances(data))
    return mean_distance


def sym_distance_matrix(A, B, eps=1e-18, self_similarity=False):
    """
    Defines the symbolic matrix that contains the distances between the vectors of A and B
    :param A: the first data matrix
    :param B: the second data matrix
    :param self_similarity: zeros the diagonial to improve the stability
    :params eps: the minimum distance between two vectors (set to a very small number to improve stability)
    :return:
    """
    # Compute the squared distances
    AA = torch.sum(A * A, 1).view(-1, 1)
    BB = torch.sum(B * B, 1).view(1, -1)
    AB = torch.mm(A, B.transpose(0, 1))
    D = AA + BB - 2 * AB

    # Zero the diagonial
    if self_similarity:
        D = D.view(-1)
        D[::B.size(0) + 1] = 0
        D = D.view(A.size(0), B.size(0))

    # Return the square root
    D = torch.sqrt(torch.clamp(D, min=eps))

    return D


def sym_heat_similarity_matrix(X, sigma):
    """
    Defines the self similarity matrix using the heat kernel
    :param X:
    :param sigma:
    :return:
    """
    D = sym_distance_matrix(X, X, self_similarity=True)
    return torch.exp(-D ** 2 / (sigma ** 2))


def fast_distance_matrix(A, B):
    """
    PyTorch based distance calculation between matrices A and B

    :param A: the first matrix
    :param B: the second matrix
    :param use_gpu: set to True, if gpu must be used
    :return: the distance matrix
    """
    use_gpu = False
    # Use GPU if available
    if torch.cuda.device_count() > 0:
        use_gpu = True

    A = Variable(torch.from_numpy(np.float32(A)))
    B = Variable(torch.from_numpy(np.float32(B)))

    if use_gpu:
        A, B = A.cuda(), B.cuda()

    D = sym_distance_matrix(A, B)

    if use_gpu:
        D = D.cpu()

    return D.data.numpy()


def fast_heat_similarity_matrix(X, sigma):
    """
    PyTorch based similarity calculation
    :param X: the matrix with the data
    :param sigma: scaling factor
    :return: the similarity matrix
    """
    use_gpu = False
    # Use GPU if available
    if torch.cuda.device_count() > 0:
        use_gpu = True

    X = Variable(torch.from_numpy(np.float32(X)))
    sigma = Variable(torch.from_numpy(np.float32([sigma])))
    if use_gpu:
        X, sigma = X.cuda(), sigma.cuda()

    D = sym_heat_similarity_matrix(X, sigma)

    if use_gpu:
        D = D.cpu()

    return D.data.numpy()
