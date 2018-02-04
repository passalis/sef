# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sef_dr.classification import evaluate_svm
from sef_dr.datasets import load_mnist
from sef_dr.linear import LinearSEF

def supervised_reduction(method=None):
    # Load data and init seeds
    train_data, train_labels, test_data, test_labels = load_mnist(dataset_path='data')
    np.random.seed(1)
    sklearn.utils.check_random_state(1)
    n_train = 5000
    n_classes = len(np.unique(train_labels))

    if method == 'lda':
        proj = LinearDiscriminantAnalysis(n_components=n_classes - 1)
        proj.fit(train_data[:n_train, :], train_labels[:n_train])
    elif method == 's-lda':
        proj = LinearSEF(train_data.shape[1], output_dimensionality=(n_classes - 1))
        proj.cuda()
        loss = proj.fit(data=train_data[:n_train, :], target_labels=train_labels[:n_train], epochs=50,
                        target='supervised', batch_size=128, regularizer_weight=1, learning_rate=0.001, verbose=True)

    elif method == 's-lda-2x':
        # SEF output dimensions are not limited
        proj = LinearSEF(train_data.shape[1], output_dimensionality=2 * (n_classes - 1))
        proj.cuda()
        loss = proj.fit(data=train_data[:n_train, :], target_labels=train_labels[:n_train], epochs=50,
                        target='supervised', batch_size=128, regularizer_weight=1, learning_rate=0.001, verbose=True)

    acc = evaluate_svm(proj.transform(train_data[:n_train, :]), train_labels[:n_train],
                       proj.transform(test_data), test_labels)

    print("Method: ", method, " Test accuracy: ", 100 * acc, " %")


if __name__ == '__main__':
    print("LDA: ")
    supervised_reduction('lda')

    print("S-LDA: ")
    supervised_reduction('s-lda')

    print("S-LDA (2x): ")
    supervised_reduction('s-lda-2x')


