# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from .targets import sim_target_copy, sim_target_supervised, sim_target_svm, sim_target_fixed
from .similarity import mean_data_distance, sym_heat_similarity_matrix
import torch
from torch.autograd import Variable


class SEF_Base(object):
    def __init__(self, input_dimensionality, output_dimensionality, scaler='default'):
        """
        SEF_Base constuctor
        :param input_dimensionality: dimensionality of the input space
        :param output_dimensionality: dimensionality of the target space
        :param scaler: the scaler used to scale the data
        """

        self.input_dimensionality = input_dimensionality
        self.output_dimensionality = output_dimensionality

        if scaler == 'default':
            self.scaler = StandardScaler()
        elif scaler is not None:
            self.scaler = scaler()
        else:
            self.scaler = None

        # Scaling factor for computing the similarity matrix of the projected data
        self.sigma_projection = np.float32(0.1)
        self.use_gpu = False

        # The parameters of the model that we want to learn
        self.trainable_params = []

        # Other non-trainable parametsr
        self.non_trainable_params = []

    def transform(self, data, batch_size=128):
        """
        Projects the input data into the lower dimensional space
        :param data: the original data
        :param batch_size: the batch size used to transform the data (allows for transforming bigger datasets at once)
        :return: the transformed data
        """

        # Scale the data
        if self.scaler is not None:
            data = np.float32(self.scaler.transform(data))
        else:
            data = np.float32(data)

        results = []

        # Process the data in batches
        n_batches = data.shape[0] // batch_size
        idx = np.arange(0, data.shape[0])

        for j in range(n_batches):
            cur_idx = idx[j * batch_size:(j + 1) * batch_size]
            # Support for h5py
            cur_idx = list(sorted(cur_idx))

            cur_data = Variable(torch.from_numpy(np.float32(data[cur_idx])))
            if self.use_gpu:
                cur_data = cur_data.cuda()
            cur_transformed_data = self._sym_project(cur_data)

            if self.use_gpu:
                cur_transformed_data = cur_transformed_data.cpu()

            results.append(cur_transformed_data.data.numpy())

        if n_batches * batch_size < data.shape[0]:
            cur_idx = idx[n_batches * batch_size:]
            cur_idx = list(sorted(cur_idx))

            cur_data = Variable(torch.from_numpy(np.float32(data[cur_idx])))
            if self.use_gpu:
                cur_data = cur_data.cuda()
            cur_transformed_data = self._sym_project(cur_data)

            if self.use_gpu:
                cur_transformed_data = cur_transformed_data.cpu()
            results.append(cur_transformed_data.data.numpy())

        results = np.concatenate(results)

        # # Wrap the data and transform
        # data = Variable(torch.from_numpy(np.float32(data)))
        # if self.use_gpu:
        #     data = data.cuda()
        # results = self._sym_project(data)
        # if self.use_gpu:
        #     results = results.cpu()
        # results  = results.data.numpy()

        return results

    def fit_transform(self, data, epochs, batch_size=128, verbose=False, target='copy', target_data=None,
                      target_labels=None, target_sigma=None, target_params={}, warm_start=False,
                      learning_rate=0.0001, regularizer_weight=0.001):
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
        :param learning_rate: the learning rate to be used for the optimization
        :param regularizer_weight: the weight of the used regularizer (real number beetween 0 to 1)
        :return:
        """
        self.fit(data, epochs, batch_size, verbose, target, target_data, target_labels, target_sigma, target_params,
                 warm_start, learning_rate, regularizer_weight)
        return self.transform(data, batch_size=batch_size)

    def fit(self, data, epochs, batch_size=128, verbose=False, target='copy', target_data=None, target_labels=None,
            target_sigma=None, target_params={}, warm_start=False, learning_rate=0.0001, regularizer_weight=0.001):
        """
        Optimizes the similarity embedding
        :param data: the data used for the optimization
        :param epochs: the number of iterations to be performed
        :param target_data: the data used to calculate the target similarity matrix using the 'target' function
        :param target_labels: the labels of the data (if available and used by the 'target' function)
        :param target_sigma: the sigma to be used (if needed by the 'target function) - if not given auto-estimated -
        :param target: the function used to calculate the target similarity matrix(either string or compatible function)
        :param batch_size: the used batch size
        :param verbose: if set to True, then outputs information regarding the optimization process
        :param warm_start: if set to True, it does not initialize the embedding function (allows for
                            finetuning the embedding)
        :param learning_rate: the learning rate to be used for the optimization
        :param regularizer_weight: the weight of the used regularizer (real number beetween 0 to 1)
        :return:
        """

        if not warm_start:
            self._initialize(data)

        # Scale the data
        if self.scaler is not None:
            data = np.float32(self.scaler.transform(data))
        else:
            data = np.float32(data)


        # Keep track of the loss during the optimization
        loss = np.zeros((epochs, 1))

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

        optimizer = torch.optim.Adam(self.trainable_params, lr=learning_rate)

        for i in range(epochs):
            np.random.shuffle(idx)

            epoch_start = time.time()
            n_batches = data.shape[0] // batch_size
            cur_loss = 0
            for j in range(n_batches):
                optimizer.zero_grad()
                cur_idx = idx[j * batch_size:(j + 1) * batch_size]
                cur_idx = list(sorted(cur_idx))

                # All the target functions must follow the same call conventions to be able to easily swap between them
                G_target, G_target_mask = target_fn(target_data, target_labels, target_sigma, cur_idx, target_params)

                # Wrap th data
                cur_data = Variable(torch.from_numpy(np.float32(data[cur_idx])))
                G_target = Variable(torch.from_numpy(np.float32(G_target)))
                G_target_mask = Variable(torch.from_numpy(np.float32(G_target_mask)))

                if self.use_gpu:
                    cur_data, G_target, G_target_mask = cur_data.cuda(), G_target.cuda(), G_target_mask.cuda()

                torch_loss = (2-regularizer_weight)*self._sym_squared_loss(cur_data, G_target, G_target_mask) \
                                                   + regularizer_weight*self._regularizer()

                torch_loss.backward()
                optimizer.step()

                if self.use_gpu:
                    torch_loss = torch_loss.cpu()

                cur_loss += torch_loss.data.item()

            if n_batches * batch_size < data.shape[0]:
                optimizer.zero_grad()
                cur_idx = idx[n_batches * batch_size:]
                cur_idx = list(sorted(cur_idx))

                # All the target functions must follow the same call conventions to be able to easily swap between them
                G_target, G_target_mask = target_fn(target_data, target_labels, target_sigma, cur_idx, target_params)

                # Wrap data
                cur_data = Variable(torch.from_numpy(np.float32(data[cur_idx])))
                G_target = Variable(torch.from_numpy(np.float32(G_target)))
                G_target_mask = Variable(torch.from_numpy(np.float32(G_target_mask)))

                if self.use_gpu:
                    cur_data, G_target, G_target_mask = cur_data.cuda(), G_target.cuda(), G_target_mask.cuda()

                torch_loss = (2-regularizer_weight)*self._sym_squared_loss(cur_data, G_target, G_target_mask) \
                                                   + regularizer_weight*self._regularizer()

                torch_loss.backward()
                optimizer.step()

                if self.use_gpu:
                    torch_loss = torch_loss.cpu()

                cur_loss += torch_loss.data.item()

            epoch_end = time.time()

            loss[i] = cur_loss
            if verbose:
                print('Iteration: ', i, 'Loss: ', loss[i], " Time left: ", (epochs - i - 1) * (epoch_end - epoch_start))

        return np.squeeze(loss)

    def _sym_squared_loss(self, X, G_target, G_target_mask):
        """
        Defines the symbolic calculation of the squared loss function between the similarity matrices

        :param X: input data
        :param G_target: target similarity matrix
        :param G_target_mask: target similarity mask matrix
        :return: the symbolic loss
        """
        Y = self._sym_project(X)

        G = sym_heat_similarity_matrix(Y, self.sigma_projection)

        loss = torch.sum(((G - G_target) ** 2) * G_target_mask) / torch.sum(G_target_mask)
        return loss

    def cpu(self):
        """
        Moves the model into CPU
        :return:
        """
        for x in self.trainable_params:
            x.data = x.data.cpu()
        for x in self.non_trainable_params:
            x.data = x.data.cpu()
        self.use_gpu = False

    def cuda(self):
        """
        Moves the model into GPU
        :return:
        """
        for x in self.trainable_params:
            x.data = x.data.cuda()
        for x in self.non_trainable_params:
            x.data = x.data.cuda()
        self.use_gpu = True

    # The following functions must be implemented by the subclasses of SEF_Base
    def _regularizer(self):
        """
        Placeholder for the function that calculates the regularizer loss
        :return: the regularization_loss
        """
        return 0

    def _sym_project(self, X):
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
