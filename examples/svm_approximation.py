import numpy as np
import sklearn
from sklearn import svm, grid_search
from sklearn.preprocessing import MinMaxScaler

from classification import evaluate_svm, evaluate_ncc
from mnist import load_mnist
from sef_dr.linear import LinearSEF


def unsupervised_approximation(method='pca'):
    # Load the data and init seeds
    train_data, train_labels, test_data, test_labels = load_mnist()
    np.random.seed(1)
    sklearn.utils.check_random_state(1)
    n_train = 5000

    if method == 'svm':
        acc = evaluate_svm(train_data[:n_train, :], train_labels[:n_train], test_data, test_labels)
    elif method == 'ncc':
        acc = evaluate_ncc(train_data[:n_train, :], train_labels[:n_train], test_data, test_labels)
    elif method == 'S-SVM-A-10d' or method == 'S-SVM-A-20d':

        # Learn an SVM
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        parameters = {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
        model = grid_search.GridSearchCV(svm.SVC(max_iter=10000, decision_function_shape='ovo'), parameters, n_jobs=-1, cv=3)
        model.fit(train_data[:n_train], train_labels[:n_train])

        params = {'model': model, 'n_labels': np.unique(train_labels).shape[0], 'scaler': scaler}

        # Learn a similarity embedding
        if method == 'S-SVM-A-10d':
            dims = 10
        else:
            dims = 20
        proj = LinearSEF(train_data.shape[1], output_dimensionality=dims, learning_rate=0.001, regularizer_weight=0.001)
        proj.init(train_data[:n_train])

        loss = proj.fit(data=train_data[:n_train, :], target_data=train_data[:n_train, :],
                        target_labels=train_labels[:n_train], target='svm', target_params=params, iters=50,
                        batch_size=128, verbose=True)

        acc = evaluate_ncc(proj.transform(train_data[:n_train, :]), train_labels[:n_train],
                           proj.transform(test_data), test_labels)

    print "Method: ", method, " Test accuracy: ", 100 * acc, " %"


if __name__ == '__main__':
    # print "Evaluating baseline SVM ..."
    # unsupervised_approximation('svm')
    #
    # print "Evaluating baseline NCC"
    # unsupervised_approximation('ncc')
    #
    # print "Evaluating SVM-based analysis 10d"
    # unsupervised_approximation('S-SVM-A-10d')

    print "Evaluating SVM-based analysis 20d"
    unsupervised_approximation('S-SVM-A-20d')