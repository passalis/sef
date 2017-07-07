from sklearn import svm, grid_search
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def evaluate_svm(train_data, train_labels, test_data, test_labels, n_jobs=-1):
    """
    Evaluates a representation using a Linear SVM
    It uses 3-fold cross validation for selecting the C parameter
    :param train_data:
    :param train_labels:
    :param test_data:
    :param test_labels:
    :param n_jobs:
    :return: the test accuracy
    """

    # Scale data to 0-1
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    parameters = {'kernel': ['linear'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    svr = svm.SVC(max_iter=10000)
    clf = grid_search.GridSearchCV(svr, parameters, n_jobs=n_jobs, cv=3)
    clf.fit(train_data, train_labels)
    lin_svm_train = clf.score(train_data, train_labels)
    lin_svm_test = clf.score(test_data, test_labels)
    return lin_svm_test


def evaluate_ncc(train_data, train_labels, test_data, test_labels):
    ncc = NearestCentroid()
    ncc.fit(train_data, train_labels)
    ncc_train = ncc.score(train_data, train_labels)
    ncc_test = ncc.score(test_data, test_labels)
    return ncc_test