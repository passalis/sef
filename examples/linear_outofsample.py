# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.manifold import Isomap
from classification import evaluate_svm
from mnist import load_mnist
from sef_dr.linear import LinearSEF


def outofsample_extensions(method='linear-regression'):
    # Load the data and init seeds
    train_data, train_labels, test_data, test_labels = load_mnist()
    np.random.seed(1)
    sklearn.utils.check_random_state(1)
    n_train_samples = 5000

    # Learn a new space using Isomap
    isomap = Isomap(n_components=10, n_neighbors=20)
    train_data_isomap = np.float32(isomap.fit_transform(train_data[:n_train_samples, :]))

    if method == 'linear-regression':
        # Use linear regression to provide baseline out-of-sample extensions
        proj = LinearRegression()
        proj.fit(np.float64(train_data[:n_train_samples, :]), np.float64(train_data_isomap))
        acc = evaluate_svm(proj.predict(train_data[:n_train_samples, :]), train_labels[:n_train_samples],
                           proj.predict(test_data), test_labels)
    elif method == 'c-ISOMAP-10d' or method == 'c-ISOMAP-20d':
        # Use the SEF to provide out-of-sample extensions
        if method == 'c-ISOMAP-10d':
            proj = LinearSEF(train_data.shape[1], output_dimensionality=10, learning_rate=0.001)
        else:
            proj = LinearSEF(train_data.shape[1], output_dimensionality=20, learning_rate=0.001)
        loss = proj.fit(data=train_data[:n_train_samples, :], target_data=train_data_isomap, target='copy',
                        iters=50, batch_size=128, verbose=True)
        acc = evaluate_svm(proj.transform(train_data[:n_train_samples, :]), train_labels[:n_train_samples],
                           proj.transform(test_data), test_labels)

    print("Method: ", method, " Test accuracy: ", 100 * acc, " %")


if __name__ == '__main__':
    print("Evaluating baseline linear-regression (10d) for providing out-of-sample extensions...")
    outofsample_extensions('linear-regression')

    print("Evaluating linear SEF (10d) for providing out-of-sample extensions...")
    outofsample_extensions('c-ISOMAP-10d')

    print("Evaluating linear SEF (10d) for providing out-of-sample extensions...")
    outofsample_extensions('c-ISOMAP-20d')
