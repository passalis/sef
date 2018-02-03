# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sef_dr.utils.classification import evaluate_svm
from sef_dr.linear import LinearSEF
from sef_dr.utils.datasets import dataset_loader


def unsupervised_approximation(method=None, dataset=None):
    np.random.seed(1)
    sklearn.utils.check_random_state(1)

    dataset_path = 'data'
    train_data, train_labels, test_data, test_labels = dataset_loader(dataset_path, dataset, seed=1)

    if method == 'pca':

        # Learn a baseline pca projection
        proj = PCA(n_components=10)
        proj.fit(train_data)

    elif method == 's-pca':
        # Learn a high dimensional projection
        proj_to_copy = PCA(n_components=50)
        proj_to_copy.fit(train_data)
        target_data = np.float32(proj_to_copy.transform(train_data))

        # Approximate it using the SEF and 10 dimensions
        proj = LinearSEF(train_data.shape[1], output_dimensionality=10)
        proj.cuda()
        loss = proj.fit(data=train_data, target_data=target_data, target='copy',
                        epochs=50, batch_size=1024, verbose=False, learning_rate=0.001, regularizer_weight=1)

    # Evaluate the method
    acc = evaluate_svm(proj.transform(train_data), train_labels, proj.transform(test_data), test_labels)

    print("Method: ", method, " Test accuracy: ", 100 * acc, " %")


if __name__ == '__main__':
    for dataset in ['15scene', 'corel', 'mnist', 'yale', 'kth', '20ng']:
        print("Evaluating dataset: ", dataset)

        print("Evaluating baseline 10d PCA ...")
        unsupervised_approximation('pca', dataset=dataset)

        print("Evaluating 10d SEF mimicking 50d PCA")
        unsupervised_approximation('s-pca', dataset=dataset)

