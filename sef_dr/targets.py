# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from .similarity import fast_heat_similarity_matrix
from sklearn.metrics.pairwise import pairwise_distances


def sim_target_copy(target_data, target_labels, sigma, idx, target_params):
    """
    Sets as target to maintain the similarity between the data
    :param target_data: the target data (to mimic)
    :param target_labels: (not used)
    :param sigma: scaling factor used to calculate the similarity matrix
    :param idx: indices of the data samples to be used for the calculation of the similarity matrix
    :return: the similarity matrix and the corresponding mask
    """

    cur_data = target_data[idx]
    Gt = fast_heat_similarity_matrix(cur_data, sigma)
    Gt_mask = np.ones_like(Gt)

    return Gt, np.float32(Gt_mask)


def sim_target_supervised(target_data, target_labels, sigma, idx, target_params):
    """
    Sets as target to bring close the points of the same class, and increase the distance of the points of different
     classes
    :param target_data: (not used)
    :param target_labels: the target labels
    :param sigma: (not used)
    :param idx: indices of the data samples to be used for the calculation of the similarity matrix
    :return: the similarity matrix and the corresponding mask
    """

    cur_labels = target_labels[idx]
    N = cur_labels.shape[0]

    # The number of labels is used to weight the target similarity of samples with different labels
    # an it is calculated per batch
    N_labels = len(np.unique(cur_labels))

    # Calculate supervised matrix
    Gt = np.zeros((N, N))
    mask = np.zeros((N, N))

    if 'in_class_similarity' in target_params:
        in_class_similarity = target_params['in_class_similarity']
    else:
        in_class_similarity = 1

    if 'bewteen_class_similarity' in target_params:
        bewteen_class_similarity = target_params['bewteen_class_similarity']
    else:
        bewteen_class_similarity = 0

    for i in range(N):
        for j in range(N):
            if cur_labels[i] == cur_labels[j]:
                Gt[i, j] = in_class_similarity
                mask[i, j] = 1
            else:
                Gt[i, j] = bewteen_class_similarity
                mask[i, j] = 1.0 / (N_labels - 1)

    return np.float32(Gt), np.float32(mask)



def generate_svm_similarity_matrix(target_data, target_labels, n_labels, model, scaler):
    """
    Generates the target similarity matrix using SVM-based similarity

    :param target_data: the data
    :param target_labels: the labels of the data
    :param model: the SVM model
    :param scaler: the used scaler
    :return:
    """

    def get_ovo_indices(n_labels):
        """
        Produces a dictionary with the 1 vs. 1 mapping for the svm
        :param classA:
        :param classB:
        :return:
        """

        dic = {}
        count = 0
        for i in range(n_labels):
            for j in range(n_labels):
                if i < j:
                    dic[(i, j)] = count
                    dic[(j, i)] = count
                    count += 1
                if i == j:
                    dic[(i, j)] = count - 1

        return dic

    if scaler:
        target_data = scaler.transform(target_data)

    N = target_data.shape[0]

    # Get the 1-1 mapping to the SVMs
    ovo_indices = get_ovo_indices(n_labels)

    # Get the distances to the hyperplane
    prob = model.decision_function(target_data)
    G = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            ovo = ovo_indices[(int(target_labels[i]), int(target_labels[j]))]
            G[i, j] = np.abs(prob[i, ovo] - prob[j, ovo])

    g = np.mean(G)
    Gt = np.exp(-G / g)
    return Gt



def sim_target_svm(target_data, target_labels, sigma, idx, target_params):
    """
    Sets as target to mimic the SVM-based similarity (SVM-based analysis)
    :param target_data: target data
    :param target_labels: target labels
    :param sigma: (not used)
    :param idx: indices of the data samples to be used for the calculation of the similarity matrix
    :param target_params: expected to find the 'model', the used 'scaler' and the number of labels ('n_labels')
    :return: the similarity matrix and the corresponding mask
    """

    Gt = generate_svm_similarity_matrix(target_data[idx], target_labels[idx], target_params['n_labels'],
                                        target_params['model'], target_params['scaler'])
    Gt_mask = np.ones((len(idx), (len(idx))))
    return np.float32(Gt), np.float32(Gt_mask)

def sim_target_svm_precomputed(target_data, target_labels, sigma, idx, target_params):
    """
    Sets as target to mimic the SVM-based similarity (SVM-based analysis)
    This function uses a precomputed similarity matrix to avoid repeated SVM calls
    :param target_data: (not used)
    :param target_labels: (not used)
    :param sigma: (not used)
    :param idx: indices of the data samples to be used for the calculation of the similarity matrix
    :param target_params: expected to find the 'Gt' (precomputed similarity matrix using the generate_svm_similarity_matrix
    :return: the similarity matrix and the corresponding mask
    """

    Gt = np.float32(target_params['Gt'])
    Gt = Gt[:, idx][idx]
    Gt_mask = np.ones((len(idx), (len(idx))))
    return np.float32(Gt), np.float32(Gt_mask)

def sim_target_fixed(target_data, target_labels, sigma, idx, target_params):
    """
    Sets as target to have fixed similarity between all the training samples
    :param target_data: (not used)
    :param target_labels: (not used)
    :param sigma: not used
    :param idx: indices of the data samples to be used for the calculation of the similarity matrix
    :param target_params: expect to found the 'target_value' here
    :return: the similarity matrix and the corresponding mask
    """
    if 'target_value' not in target_params:
        target_params['target_value'] = 0.0

    Gt = np.ones((len(idx), len(idx)))
    Gt = Gt * target_params['target_value']
    Gt_mask = np.ones_like(Gt)

    return np.float32(Gt), np.float32(Gt_mask)
