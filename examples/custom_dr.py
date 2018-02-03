# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from sef_dr.utils.datasets import dataset_loader
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sef_dr import KernelSEF


def sim_target_supervised(target_data, target_labels, sigma, idx, target_params):
    cur_labels = target_labels[idx]
    N = cur_labels.shape[0]

    N_labels = len(np.unique(cur_labels))

    Gt, mask = np.zeros((N, N)), np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if cur_labels[i] == cur_labels[j]:
                Gt[i, j] = 0.8
                mask[i, j] = 1
            else:
                Gt[i, j] = 0.1
                mask[i, j] = 1.0 / (N_labels - 1)

    return np.float32(Gt), np.float32(mask)


if __name__ == '__main__':
    np.random.seed(1)

    # Load data and sample 3 classes
    train_data, train_labels, test_data, test_labels = dataset_loader(dataset='mnist')
    idx = np.logical_or(train_labels == 0, train_labels == 1)
    idx = np.logical_or(idx, train_labels == 2)
    train_data, train_labels = train_data[idx], train_labels[idx]
    train_data, train_labels = train_data[:100, :], train_labels[:100]

    # Perform DR
    proj = KernelSEF(train_data, train_data.shape[0], 2, sigma=3)
    proj.fit(train_data, target_labels=train_labels, target=sim_target_supervised, epochs=500,  learning_rate=0.0001, regularizer_weight=0,
             verbose=True)
    train_data = proj.transform(train_data)

    # Plot the results
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels,
                cmap=matplotlib.colors.ListedColormap(['red', 'green', 'blue', ]))
    plt.legend(handles=[mpatches.Patch(color='red', label='Digit 0'), mpatches.Patch(color='green', label='Digit 1'),
                        mpatches.Patch(color='blue', label='Digit 2')], loc='upper left')

    plt.savefig('custom_dr.png', bbox_inches='tight')
    plt.show()
