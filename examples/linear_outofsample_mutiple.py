# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.manifold import Isomap
from classification import evaluate_svm
from datasets import dataset_loader
from sef_dr.linear import LinearSEF


def outofsample_extensions(method=None, dataset=None):
    np.random.seed(1)
    sklearn.utils.check_random_state(1)

    train_data, train_labels, test_data, test_labels = dataset_loader(dataset, seed=1)

    # Learn a new space using Isomap
    isomap = Isomap(n_components=10, n_neighbors=20)
    train_data_isomap = np.float32(isomap.fit_transform(train_data))

    if method == 'linear-regression':
        from sklearn.preprocessing import StandardScaler
        std = StandardScaler()
        train_data = std.fit_transform(train_data)
        test_data = std.transform(test_data)

        # Use linear regression to provide baseline out-of-sample extensions
        proj = LinearRegression()
        proj.fit(np.float64(train_data), np.float64(train_data_isomap))
        acc = evaluate_svm(proj.predict(train_data), train_labels,
                           proj.predict(test_data), test_labels)
    elif method == 'c-ISOMAP-10d' or method == 'c-ISOMAP-20d':
        # Use the SEF to provide out-of-sample extensions
        if method == 'c-ISOMAP-10d':
            proj = LinearSEF(train_data.shape[1], output_dimensionality=10)
            proj.cuda()
        else:
            proj = LinearSEF(train_data.shape[1], output_dimensionality=20)
            proj.cuda()
        loss = proj.fit(data=train_data, target_data=train_data_isomap, target='copy',
                        epochs=50, batch_size=1024, verbose=False, learning_rate=0.001, regularizer_weight=1)
        acc = evaluate_svm(proj.transform(train_data), train_labels,
                           proj.transform(test_data), test_labels)

    print("Method: ", method, " Test accuracy: ", 100 * acc, " %")


if __name__ == '__main__':

    for dataset in ['15scene', 'corel', 'mnist', 'yale', 'kth', '20ng']:
        print("Evaluating dataset: ", dataset)

        print("Evaluating baseline linear-regression (10d) for providing out-of-sample extensions...")
        outofsample_extensions('linear-regression', dataset=dataset)

        print("Evaluating linear SEF (10d) for providing out-of-sample extensions...")
        outofsample_extensions('c-ISOMAP-10d', dataset=dataset)

        print("Evaluating linear SEF (10d) for providing out-of-sample extensions...")
        outofsample_extensions('c-ISOMAP-20d', dataset=dataset)
