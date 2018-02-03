# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import sklearn
from sklearn import svm, grid_search
from classification import evaluate_svm, evaluate_ncc
from datasets import dataset_loader
from sef_dr.targets import generate_svm_similarity_matrix, sim_target_svm_precomputed
from sef_dr.linear import LinearSEF
from sklearn.preprocessing import LabelEncoder


def svm_approximation(method=None, dataset=None):
    np.random.seed(1)
    sklearn.utils.check_random_state(1)

    dataset_path = 'data'
    train_data, train_labels, test_data, test_labels = dataset_loader(dataset_path, dataset, seed=1)

    lab = LabelEncoder()
    train_labels = lab.fit_transform(train_labels)
    test_labels = lab.transform(test_labels)

    if method == 'svm':
        acc = evaluate_svm(train_data, train_labels, test_data, test_labels)
    elif method == 'ncc':

        acc = evaluate_ncc(train_data, train_labels, test_data, test_labels)
    elif method == 'S-SVM-A-10d' or method == 'S-SVM-A-20d':

        # Learn an SVM
        parameters = {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
        model = grid_search.GridSearchCV(svm.SVC(max_iter=10000, decision_function_shape='ovo'), parameters, n_jobs=-1,
                                         cv=3)
        model.fit(train_data, train_labels)

        # params = {'model': model, 'n_labels': np.unique(train_labels).shape[0], 'scaler': None}
        Gt = generate_svm_similarity_matrix(train_data, train_labels,
                                            len(np.unique(train_labels)), model, None)
        params = {'Gt': Gt}

        # Learn a similarity embedding
        if method == 'S-SVM-A-10d':
            dims = len(np.unique(train_labels))
        else:
            dims = 2*len(np.unique(train_labels))

        proj = LinearSEF(train_data.shape[1], output_dimensionality=dims)
        proj.cuda()
        loss = proj.fit(data=train_data, target_data=train_data, target_labels=train_labels,
                        target=sim_target_svm_precomputed, target_params=params, epochs=100, learning_rate=0.001,
                        batch_size=256, verbose=True, regularizer_weight=0.001)

        acc = evaluate_ncc(proj.transform(train_data), train_labels,
                           proj.transform(test_data), test_labels)

    print("Method: ", method, " Test accuracy: ", 100 * acc, " %")


if __name__ == '__main__':

    for dataset in ['15scene', 'corel', 'mnist', 'yale', 'kth', '20ng']:
        print("Evaluating dataset: ", dataset)

        print("Evaluating baseline SVM ...")
        svm_approximation('svm', dataset=dataset)

        print("Evaluating baseline NCC")
        svm_approximation('ncc', dataset=dataset)

        print("Evaluating SVM-based analysis 10d")
        svm_approximation('S-SVM-A-10d', dataset=dataset)

        print("Evaluating SVM-based analysis 20d")
        svm_approximation('S-SVM-A-20d', dataset=dataset)
