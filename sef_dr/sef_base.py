import time
import numpy as np
import theano
import theano.tensor as T
from sklearn.preprocessing import StandardScaler
from targets import sim_target_copy, sim_target_supervised, sim_target_svm, sim_target_fixed
from sef_dr.similarity import mean_data_distance, sym_similarity_matrix


class SEF_Base(object):
    def __init__(self, input_dimensionality, output_dimensionality, learning_rate, scaler=None):
        """
        SEF_Base constuctor
        :param input_dimensionality: dimensionality of the input space
        :param output_dimensionality: dimensionality of the target space
        :param learning_rate: the learning rate used for the optimization 
        :param scaler: the scaler used to scale the data
        """

        self.input_dimensionality = input_dimensionality
        self.output_dimensionality = output_dimensionality
        self.learning_rate = theano.shared(np.float32(learning_rate))

        # Input data variables
        self.X = T.matrix('X')
        self.Gt = T.matrix('Gt')
        self.Gt_mask = T.matrix('Gt_mask')

        if scaler == 'default':
            self.scaler = StandardScaler()
        elif scaler is not None:
            self.scaler = scaler()
        else:
            self.scaler = None

        # Theano functions placeholder
        self.train_fn = None
        self.project_fn = None

        # Scaling factor for computing the similarity matrix of the projected data
        self.sigma_projection = theano.shared(value=np.float32(0.1), name='sigma_projection')

    def transform(self, data):
        """
        Projects the input data into the lower dimensional space
        :param data: the original data
        :return: the transformed data
        """
        # Scale the data
        if self.scaler is not None:
            data = np.float32(self.scaler.transform(data))
        else:
            data = np.float32(data)

        # Project the scaled data
        return self.project_fn(data)

    def fit_transform(self, data, iters, batch_size=128, verbose=False, target='copy', target_data=None,
                      target_labels=None,
                      target_sigma=None, target_params={}, warm_start=False):
        """
        Optimizes the similarity embedding and returns the projected data
        :param data: the data used for the optimization
        :param iters: the number of iterations to be performed
        :param target_data: the data used to calculate the target similarity matrix using the 'target' function
        :param target_labels: the labels of the data (if available and used by the 'target' function)
        :param target_sigma: the sigma to be used (if needed by the 'target function) - if not given auto-estimated -
        :param target: the function used to calculate the target similarity matrix(either string or compatible function)
        :param batch_size: the used batch size
        :param verbose: if set to True, then outputs information regarding the optimization process 
        :param warm_start: if set to True, does not initialize the embedding function
        :return: 
        """
        self.fit(data, iters, batch_size, verbose, target, target_data, target_labels, target_sigma, target_params,
                 warm_start)
        return self.transform(data)

    def fit(self, data, iters, batch_size=128, verbose=False, target='copy', target_data=None, target_labels=None,
            target_sigma=None, target_params={}, warm_start=False):
        """
        Optimizes the similarity embedding
        :param data: the data used for the optimization
        :param iters: the number of iterations to be performed
        :param target_data: the data used to calculate the target similarity matrix using the 'target' function
        :param target_labels: the labels of the data (if available and used by the 'target' function)
        :param target_sigma: the sigma to be used (if needed by the 'target function) - if not given auto-estimated -
        :param target: the function used to calculate the target similarity matrix(either string or compatible function)
        :param batch_size: the used batch size
        :param verbose: if set to True, then outputs information regarding the optimization process 
        :param warm_start: if set to True, does not initialize the embedding function
        :return: 
        """

        if not warm_start:
            self._initialize(data)

        # Keep track of the loss during the optimization
        loss = np.zeros((iters, 1))

        # If batch size not selected, set to whole dataset
        if batch_size is None or batch_size <= 0:
            batch_size = data.shape[0]

        # Use the selected method for learning the similarity embedding
        if target == 'copy':
            if target_sigma is None or target_sigma == 0:
                target_sigma = np.float32(mean_data_distance(target_data))
            target_fn = sim_target_copy
        elif target == 'supervised':
            target_fn = sim_target_supervised
        elif target == 'svm':
            target_fn = sim_target_svm
        elif target == 'fixed':
            target_fn = sim_target_fixed
        else:
            target_fn = target

        # Optimize the embedding
        idx = np.arange(data.shape[0])

        for i in range(iters):
            np.random.shuffle(idx)

            epoch_start = time.time()
            n_batches = data.shape[0] // batch_size
            cur_loss = 0
            for j in range(n_batches):
                cur_idx = idx[j * batch_size:(j + 1) * batch_size]
                # Support for h5py
                cur_idx = list(sorted(cur_idx))
                # All the target functions must follow the same call conventions to be able to easily swap between them
                G_target, G_target_mask = target_fn(target_data, target_labels, target_sigma, cur_idx, target_params)
                cur_loss += self.train_fn(data[cur_idx], G_target, G_target_mask)
            if n_batches * batch_size < data.shape[0]:
                cur_idx = idx[n_batches * batch_size:]
                cur_idx.sort()
                G_target, G_target_mask = target_fn(target_data, target_labels, target_sigma, cur_idx, target_params)
                cur_loss += self.train_fn(data[cur_idx], G_target, G_target_mask)
            epoch_end = time.time()

            loss[i] = cur_loss
            if verbose:
                print 'Iteration: ', i, 'Loss: ', loss[i], " Time left: ", (iters - i - 1) * (epoch_end - epoch_start)

        return np.squeeze(loss)

    def _sym_project_data(self, X):
        """
        Placeholder for symbolic project data function
        :param X: a symbolic variable holding the input data
        :return: a symbolic variable of the projected data
        """
        return X

    def _initialize(self, data):
        """
        Placeholder for initializer
        :param data: the data used for initialized the model
        :return: 
        """
        pass

    def _sym_loss(self, X, G_target, G_target_mask):
        """
        Defines the symbolic calculation of the squared loss function between the similarity matrices

        :param X: input data 
        :param G_target: target similarity matrix
        :param G_target_mask: target similarity mask matrix
        :return: the symbolic loss
        """
        Y = self._sym_project_data(X)
        G = sym_similarity_matrix(Y, self.sigma_projection)
        loss = T.sum(((G - G_target) ** 2) * G_target_mask) / T.sum(G_target_mask)
        return loss
