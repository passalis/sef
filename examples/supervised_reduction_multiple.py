# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sef_dr.classification import evaluate_svm
from sef_dr.linear import LinearSEF
from sef_dr.datasets import dataset_loader
from sklearn.preprocessing import StandardScaler


def supervised_reduction(method=None, dataset=None):
    np.random.seed(1)
    sklearn.utils.check_random_state(1)

    dataset_path = 'data'
    train_data, train_labels, test_data, test_labels = dataset_loader(dataset_path, dataset, seed=1)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    if dataset == 'yale':
        regularizer_weight = 0.0001
    else:
        regularizer_weight = 1

    n_classes = len(np.unique(train_labels))

    if method == 'lda':
        proj = LinearDiscriminantAnalysis(n_components=n_classes - 1)
        proj.fit(train_data, train_labels)
    elif method == 's-lda':
        proj = LinearSEF(train_data.shape[1], output_dimensionality=(n_classes - 1))
        proj.cuda()
        loss = proj.fit(data=train_data, target_labels=train_labels, epochs=100,
                        target='supervised', batch_size=256, regularizer_weight=regularizer_weight, learning_rate=0.001,
                        verbose=False)

    elif method == 's-lda-2x':
        # SEF output dimensions are not limited
        proj = LinearSEF(train_data.shape[1], output_dimensionality=2 * (n_classes - 1))
        proj.cuda()
        loss = proj.fit(data=train_data, target_labels=train_labels, epochs=100,
                        target='supervised', batch_size=256, regularizer_weight=regularizer_weight, learning_rate=0.001,
                        verbose=False)

    acc = evaluate_svm(proj.transform(train_data), train_labels,
                       proj.transform(test_data), test_labels)

    print("Method: ", method, " Test accuracy: ", 100 * acc, " %")


if __name__ == '__main__':

    for dataset in ['15scene', 'corel', 'mnist', 'yale', 'kth', '20ng']:
        print("Evaluating dataset: ", dataset)

        print("LDA: ")
        supervised_reduction('lda', dataset=dataset)

        print("S-LDA: ")
        supervised_reduction('s-lda', dataset=dataset)

        print("S-LDA (2x): ")
        supervised_reduction('s-lda-2x', dataset=dataset)
