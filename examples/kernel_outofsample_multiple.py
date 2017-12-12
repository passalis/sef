# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import sklearn
from sklearn.kernel_ridge import KernelRidge
from sklearn.manifold import Isomap
from classification import evaluate_svm
from datasets import dataset_loader
from sef_dr.kernel import KernelSEF
from sef_dr.sef_base import mean_data_distance



def outofsample_extensions(method=None, dataset=None):
    np.random.seed(1)
    sklearn.utils.check_random_state(1)

    train_data, train_labels, test_data, test_labels = dataset_loader(dataset, seed=1)
    # Learn a new space using Isomap
    isomap = Isomap(n_components=10, n_neighbors=20)
    train_data_isomap = np.float32(isomap.fit_transform(train_data))
    sigma = mean_data_distance(np.float32(train_data))

    if method == 'kernel-regression':
        # Use kernel regression to provide baseline out-of-sample extensions
        proj = KernelRidge(kernel='rbf', gamma=(1.0 / sigma ** 2))
        proj.fit(np.float64(train_data), np.float64(train_data_isomap))
        acc = evaluate_svm(proj.predict(train_data), train_labels,
                           proj.predict(test_data), test_labels)
    elif method == 'cK-ISOMAP-10d' or method == 'cK-ISOMAP-20d':
        # Use the SEF to provide out-of-sample extensions
        if method == 'cK-ISOMAP-10d':
            dims = 10
        else:
            dims = 20
        proj = KernelSEF(train_data, train_data.shape[1], output_dimensionality=dims)
        proj.cuda()
        loss = proj.fit(data=train_data, target_data=train_data_isomap, target='copy',
                        epochs=100, batch_size=256, verbose=True, learning_rate=0.00001,  regularizer_weight=0.001)
        acc = evaluate_svm(proj.transform(train_data), train_labels,
                           proj.transform(test_data), test_labels)

    print("Method: ", method, " Test accuracy: ", 100 * acc, " %")


if __name__ == '__main__':


    for dataset in ['15scene', 'corel', 'mnist', 'yale', 'kth', '20ng']:
        print("Evaluating dataset: ", dataset)

        print("Evaluating baseline kernel-regression (10d) for providing out-of-sample extensions...")
        outofsample_extensions('kernel-regression', dataset=dataset)

        print("Evaluating kernel SEF (10d) for providing out-of-sample extensions...")
        outofsample_extensions('cK-ISOMAP-10d', dataset=dataset)

        print("Evaluating kernel SEF (20d) for providing out-of-sample extensions...")
        outofsample_extensions('cK-ISOMAP-20d', dataset=dataset)

