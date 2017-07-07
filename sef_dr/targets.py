import numpy as np
from sef_dr.utils.similarity import fast_similarity_matrix_fn
from sklearn.preprocessing import LabelEncoder


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
    Gt = fast_similarity_matrix_fn(cur_data, sigma)
    Gt_mask = np.ones_like(Gt)

    return Gt, np.float32(Gt_mask)


def sim_target_supervised(target_data, target_labels, sigma, idx, target_params):
    """
    Sets as target to bring close the points of the same class, and increase the distance of the points of different
     classes
    :param target_data:
    :param target_labels:
    :param sigma:
    :param idx:
    :return:
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


def sim_target_svm(target_data, target_labels, sigma, idx, target_params):
    """
    Sets as target to mimic the SVM-based similarity (SVM-based analysis)
    :param target_data:
    :param target_labels:
    :param sigma:
    :param idx:
    :param target_params:
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

    target_data = target_params['scaler'].transform(target_data[idx])
    target_labels = target_labels[idx]
    model = target_params['model']
    n_labels = target_params['n_labels']
    N = target_data.shape[0]

    # Get the 1-1 mapping to the SVMs
    ovo_indices = get_ovo_indices(n_labels)

    # Get the distances to the hyperplane
    prob = model.decision_function(target_data)
    G = np.zeros((N, N))
    Gt_mask = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            ovo = ovo_indices[(int(target_labels[i]), int(target_labels[j]))]
            G[i, j] = np.abs(prob[i, ovo] - prob[j, ovo])

    g = np.mean(G)
    Gt = np.exp(-G / (g ** 2))
    return np.float32(Gt), np.float32(Gt_mask)