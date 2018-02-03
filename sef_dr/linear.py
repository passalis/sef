# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from sklearn.decomposition import PCA
from .sef_base import SEF_Base
from .similarity import mean_data_distance

import torch
from torch.autograd import Variable


class LinearSEF(SEF_Base):
    def __init__(self, input_dimensionality, output_dimensionality, scaler='default'):
        """
        Creats a Linear SEF object
        :param input_dimensionality: dimensionality of the input space
        :param output_dimensionality: dimensionality of the target space
        :param learning_rate: learning rate to be used for the optimization
        :param regularizer_weight: the weight of the regularizer
        :param scaler:
        """

        # Call base constructor
        SEF_Base.__init__(self, input_dimensionality, output_dimensionality, scaler)

        # Projection weights variables
        W = np.float32(0.1 * np.random.randn(self.input_dimensionality, output_dimensionality))
        self.W = Variable(torch.from_numpy(W), requires_grad=True)
        self.trainable_params = [self.W]

    def _initialize(self, data, n_samples=5000):
        """
        Initializes the linear SEF model
        :param data: Data to be used for the initialization
        :param n_samples: Number of samples to be used for initializing the model
        :return:
        """
        # Subsample the data
        idx = np.random.permutation(data.shape[0])[:n_samples]
        data = data[idx]

        original_data = data
        # Initialize values
        if self.scaler is None:
            data = np.float32(data)
        else:
            data = np.float32(self.scaler.fit_transform(data))

        # Use pca for initialization
        pca = PCA(n_components=self.output_dimensionality)
        pca.fit_transform(data)

        if self.use_gpu:
            self.W.data = torch.from_numpy(np.float32(pca.components_.transpose())).cuda()
        else:
            self.W.data = torch.from_numpy(np.float32(pca.components_.transpose()))

        # Estimate the sigma projection value (this usually makes the optimization "easier")
        # This can be pre-set to scale the projection into a specific range
        sigma = mean_data_distance(self.transform(original_data))
        self.sigma_projection = np.float32(sigma)

    def _regularizer(self):

        if self.use_gpu:
            regularizer = torch.mm(self.W.transpose(0, 1), self.W) - Variable(torch.eye(self.W.size(1)).cuda())
        else:
            regularizer = torch.mm(self.W.transpose(0, 1), self.W) - Variable(torch.eye(self.W.size(1)))
        return 0.5 * torch.sum(regularizer ** 2) / (self.W.size(1) ** 2)

    def _sym_project(self, X):
        return torch.mm(X, self.W)
