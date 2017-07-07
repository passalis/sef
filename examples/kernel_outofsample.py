import numpy as np
import sklearn
from sklearn.kernel_ridge import KernelRidge
from sklearn.manifold import Isomap

from classification import evaluate_svm
from mnist import load_mnist
from sef_dr.kernel import KernelSEF
from sef_dr.sef_base import mean_data_distance


def outofsample_extensions(method='kernel-regression'):
    # Load the data and init seeds
    train_data, train_labels, test_data, test_labels = load_mnist()
    np.random.seed(1)
    sklearn.utils.check_random_state(1)
    n_train_samples = 5000

    # Learn a new space using Isomap
    isomap = Isomap(n_components=10, n_neighbors=20)
    train_data_isomap = np.float32(isomap.fit_transform(train_data[:n_train_samples, :]))
    sigma = mean_data_distance(np.float32(train_data[:n_train_samples, :]))
    print "Sigma = ", sigma
    if method == 'kernel-regression':
        # Use kernel regression to provide baseline out-of-sample extensions
        proj = KernelRidge(kernel='rbf', gamma=(1.0 / sigma ** 2))
        proj.fit(np.float64(train_data[:n_train_samples, :]), np.float64(train_data_isomap))
        acc = evaluate_svm(proj.predict(train_data[:n_train_samples, :]), train_labels[:n_train_samples],
                           proj.predict(test_data), test_labels)
    elif method == 'cK-ISOMAP-10d' or method == 'cK-ISOMAP-20d':
        # Use the SEF to provide out-of-sample extensions
        if method == 'cK-ISOMAP-10d':
            dims = 10
        else:
            dims = 20

        proj = KernelSEF(train_data[:n_train_samples], train_data.shape[1], output_dimensionality=dims,
                         learning_rate=0.0001, regularizer_weight=0.001)
        proj.init(train_data[:n_train_samples, :])

        loss = proj.fit(data=train_data[:n_train_samples, :], target_data=train_data_isomap, target='copy',
                        iters=50, batch_size=128, verbose=True)
        acc = evaluate_svm(proj.transform(train_data[:n_train_samples, :]), train_labels[:n_train_samples],
                           proj.transform(test_data), test_labels)

    print "Method: ", method, " Test accuracy: ", 100 * acc, " %"


if __name__ == '__main__':
    print "Evaluating baseline kernel-regression (10d) for providing out-of-sample extensions..."
    outofsample_extensions('kernel-regression')

    print "Evaluating kernel SEF (10d) for providing out-of-sample extensions..."
    outofsample_extensions('cK-ISOMAP-10d')

    print "Evaluating kernel SEF (20d) for providing out-of-sample extensions..."
    outofsample_extensions('cK-ISOMAP-20d')
