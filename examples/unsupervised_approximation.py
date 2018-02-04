# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sef_dr.classification import evaluate_svm
from sef_dr.datasets import load_mnist
from sef_dr.linear import LinearSEF


def unsupervised_approximation(method='pca'):
    # Load the data and init seeds
    train_data, train_labels, test_data, test_labels = load_mnist(dataset_path='data')
    np.random.seed(1)
    sklearn.utils.check_random_state(1)
    n_train_samples = 5000

    if method == 'pca':

        # Learn a baseline pca projection
        proj = PCA(n_components=10)
        proj.fit(train_data[:n_train_samples, :])

    elif method == 's-pca':

        # Learn a high dimensional projection
        proj_to_copy = PCA(n_components=50)
        proj_to_copy.fit(train_data[:n_train_samples, :])
        target_data = np.float32(proj_to_copy.transform(train_data[:n_train_samples, :]))

        # Approximate it using the SEF and 10 dimensions
        proj = LinearSEF(train_data.shape[1], output_dimensionality=10)
        proj.cuda()
        loss = proj.fit(data=train_data[:n_train_samples, :], target_data=target_data, target='copy',
                        epochs=50, batch_size=128, verbose=True, learning_rate=0.001, regularizer_weight=1)


    # Evaluate the method
    acc = evaluate_svm(proj.transform(train_data[:n_train_samples, :]), train_labels[:n_train_samples],
                       proj.transform(test_data), test_labels)

    print("Method: ", method, " Test accuracy: ", 100 * acc, " %")


if __name__ == '__main__':
    print("Evaluating baseline 10d PCA ...")
    unsupervised_approximation('pca')

    print("Evaluating 10d SEF mimicking 50d PCA")
    unsupervised_approximation('s-pca')

