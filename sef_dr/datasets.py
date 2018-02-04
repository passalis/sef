from sklearn.datasets import fetch_20newsgroups
import torchvision
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import join
import numpy as np
import pickle

#TODO: Update mnist examples!!!

def dataset_loader(dataset_path=None, dataset='mnist', seed=1):
    """
    Loads a dataset and creates the appropriate train/test splits
    :param dataset_path: path to the datasets
    :param dataset: the dataset to be loaded
    :param seed: seed for creating train/test splits (when appropriate)
    :return:
    """

    np.random.seed(seed)
    n_train = 5000

    if dataset == 'mnist':
        train_data, train_labels, test_data, test_labels = load_mnist(dataset_path)
        train_data = train_data[:n_train, :]
        train_labels = train_labels[:n_train]
    elif dataset == '20ng':
        train_data, train_labels, test_data, test_labels = load_20ng_dataset_bow()
        train_data = train_data[:n_train, :]
        train_labels = train_labels[:n_train]
    elif dataset == '15scene':
        data, labels = load_15_scene_bow_features(dataset_path)
        # Get a split with 100 samples per class
        train_idx, test_idx = [], []
        idx = np.random.permutation(data.shape[0])
        data = data[idx]
        labels = labels[idx]
        for i in range(15):
            class_idx = np.where(labels == i)[0]
            train_idx.append(class_idx[:100])
            test_idx.append(class_idx[100:])
        train_idx = np.int64(np.concatenate(train_idx))
        test_idx = np.int64(np.concatenate(test_idx))
        # Get the actual split
        train_data = data[train_idx, :]
        train_labels = labels[train_idx]
        test_data = data[test_idx, :]
        test_labels = labels[test_idx]

    elif dataset == 'corel':
        data, labels = load_corel_bow_features(dataset_path)
        idx = np.random.permutation(data.shape[0])
        data = data[idx]
        labels = labels[idx]

        # Get the actual split
        train_data = data[:4800, :]
        train_labels = labels[:4800]
        test_data = data[4800:, :]
        test_labels = labels[4800:]
    elif dataset == 'yale':
        data, labels = load_yale_dataset(dataset_path)

        # Get a split with 30 per sample
        train_idx, test_idx = [], []
        idx = np.random.permutation(data.shape[0])
        data = data[idx]
        labels = labels[idx]
        for i in range(38):
            class_idx = np.where(labels == i)[0]
            train_idx.append(class_idx[:30])
            test_idx.append(class_idx[30:])
        train_idx = np.int64(np.concatenate(train_idx))
        test_idx = np.int64(np.concatenate(test_idx))

        # Get the actual split
        train_data = data[train_idx, :]
        train_labels = labels[train_idx]
        test_data = data[test_idx, :]
        test_labels = labels[test_idx]

    elif dataset == 'kth':
        train_data, train_labels, test_data, test_labels = load_kth(dataset_path)
        idx = np.random.permutation(train_data.shape[0])
        train_data, train_labels = train_data[idx], train_labels[idx]
        idx = np.random.permutation(test_data.shape[0])
        test_data, test_labels = test_data[idx], test_labels[idx]
    else:
        print("Unknown dataset!")
        assert False

    return train_data, train_labels, test_data, test_labels


def load_mnist(dataset_path):
    """
    Loads the MNIST dataset
    :return:
    """

    # Get the train split
    mnist = torchvision.datasets.MNIST(root=dataset_path, download=True, train=True)
    x_train, y_train = mnist.train_data.numpy(), mnist.train_labels.numpy()

    # Get the test split
    mnist = torchvision.datasets.MNIST(root=dataset_path, download=True, train=False)
    x_test, y_test = mnist.test_data.numpy(), mnist.test_labels.numpy()

    x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0
    x_test = x_test.reshape((x_test.shape[0], -1)) / 255.0

    return np.float32(x_train), y_train, np.float32(x_test), y_test

def load_20ng_dataset_bow():
    """
    Loads the 20NG dataset
    :return:
    """

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')

    # Convert data to tf-idf

    vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.95)
    train_data = vectorizer.fit_transform(newsgroups_train.data)
    test_data = vectorizer.transform(newsgroups_test.data)
    train_data = train_data.todense()
    test_data = test_data.todense()
    train_labels = newsgroups_train.target
    test_labels = newsgroups_test.target

    return train_data, train_labels, test_data, test_labels


def load_15_scene_bow_features(datasets_path):
    """
    Loads the pre-extracted BoF features for the 15-scene dataset
    :return:
    """
    datafile = join(datasets_path, 'scenes.pickle')
    with open(datafile, 'rb') as f:
        features = pickle.load(f, encoding='latin1')
        labels = pickle.load(f, encoding='latin1')
    labels = np.asarray(np.squeeze(labels), dtype='int')
    features = np.asarray(features, dtype='float32')
    return features, labels


def load_corel_bow_features(datasets_path):
    """
    Loads the pre-extracted BoF features for the Corel dataset
    :return:
    """
    datafile = join(datasets_path, 'corel.pickle')

    with open(datafile, 'rb') as f:
        features = pickle.load(f, encoding='latin1')
        labels = pickle.load(f, encoding='latin1')
    labels = np.asarray(np.squeeze(labels), dtype='int')
    features = np.asarray(features, dtype='float32')

    return features, labels


def load_kth(datasets_path):
    """
    Loads the HoF/HoG features for the KTH dataset
    :return:
    """
    datafile = join(datasets_path, 'kth.pickle')
    with open(datafile, 'rb') as f:
        train_data, train_labels = pickle.load(f, encoding='latin1'), pickle.load(f, encoding='latin1')
        test_data, test_labels = pickle.load(f, encoding='latin1'), pickle.load(f, encoding='latin1')

    return train_data, train_labels, test_data, test_labels


def load_yale_dataset(datasets_path):
    """
    Loads the ORL dataset
    """
    datafile = join(datasets_path, 'yale.pickle')

    with open(datafile, 'rb') as f:
        features = pickle.load(f)
        labels = pickle.load(f)

    features = [x for x in features]
    features = np.asarray(features) / 255.0
    labels = np.asarray(np.squeeze(labels), dtype='int')

    return features, labels

