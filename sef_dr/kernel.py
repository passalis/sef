# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from sklearn.decomposition import KernelPCA
from .similarity import sym_distance_matrix, mean_data_distance
from .sef_base import SEF_Base

import torch
from torch.autograd import Variable


class KernelSEF(SEF_Base):
    def __init__(self, data, input_dimensionality, output_dimensionality, kernel_type='rbf',
                 degree=2, sigma=0, kernel_scaling=1, c=1, scaler=None):
        """
        Creates a Kernel SEF object
        :param data: the data to be used by the kernel
        :param input_dimensionality: dimensionality of the input space
        :param output_dimensionality: dimensionality of the target space
        :param learning_rate: learning rate to be used for the optimization
        :param kernel_type: supported kernel: 'rbf', 'poly', and 'linear'
        :param degree: degree of the polynomial kernel
        :param sigma: the sigma value for the RBF kernel
        :param kernel_scaling: scaling parameter for the kernel
        :param c: constant kernel param for linear and poly kernels
        :param regularizer_weight: weight of the regularizer
        :param scaler: the sklearn-compatible scaler (or None)
        """
        # Call base constructor
        SEF_Base.__init__(self, input_dimensionality, output_dimensionality, scaler=scaler)

        # Adjustable parameters
        self.kernel_type = kernel_type
        self.degree = degree
        self.sigma_kernel = np.float32(sigma)
        self.alpha = kernel_scaling
        self.c = c

        # If scaler is used, fit it!
        if self.scaler is None:
            data = np.float32(data)
        else:
            pass
            data = np.float32(self.scaler.fit_transform(data))

        # If the rbf kernel is used and no sigma is supplied, estimate it!
        if sigma == 0 and self.kernel_type == 'rbf':
            sigma_kernel = np.float32(mean_data_distance(data))
            self.sigma_kernel = sigma_kernel
        else:
            self.sigma_kernel = sigma

        # Use kPCA for initialization
        kpca = KernelPCA(kernel=self.kernel_type, n_components=self.output_dimensionality,
                         gamma=(1.0 / (self.sigma_kernel ** 2)), degree=self.degree, eigen_solver='dense')
        kpca.fit(data)
        A = kpca.alphas_
        # Scale the coefficients to have unit norm (avoid rescaling)
        A = A / np.sqrt(np.diag(np.dot(A.T, np.dot(np.dot(data, data.T), A))))

        # Model parameters
        self.X_kernel = Variable(torch.from_numpy(np.float32(data)), requires_grad=False)
        self.A = Variable(torch.from_numpy(np.float32(A)), requires_grad=True)

        self.trainable_params = [self.A]
        self.non_trainable_params = [self.X_kernel]

    def _initialize(self, data, n_samples=5000):
        """
        Initializes the kernel SEF model
        :param data: Data to be used for the initialization
        :param n_samples: Number of samples to be used for initializing the model
        :return:
        """
        # Subsample the data
        idx = np.random.permutation(data.shape[0])[:n_samples]
        data = data[idx]

        # Estimate the sigma projection value (this usually makes the optimization "easier")
        self.sigma_projection = np.float32(mean_data_distance(self.transform(data)))

    def symbolic_kernel(self, X):
        if self.kernel_type == 'linear':
            K = self.alpha * torch.dot(X, self.X_kernel.transpose(0, 1)) + self.c
        elif self.kernel_type == 'poly':
            K = (self.alpha * torch.dot(X, self.X_kernel.transpose(0, 1)) + self.c) ** self.degree
        elif self.kernel_type == 'rbf':
            D = sym_distance_matrix(X, self.X_kernel, self_similarity=False)
            K = torch.exp(-D ** 2 / (self.sigma_kernel ** 2))
        else:
            raise Exception('Unknown kernel type: ', self.kernel_type)
        return K

    def _regularizer(self):
        if self.use_gpu:
            regularizer = torch.mm(self.A.transpose(0, 1), torch.mm(self.symbolic_kernel(self.X_kernel), self.A)) \
                          - Variable(torch.eye(self.A.size(1)).cuda())
        else:
            regularizer = torch.mm(self.A.transpose(0, 1), torch.mm(self.symbolic_kernel(self.X_kernel), self.A)) \
                          - Variable(torch.eye(self.A.size(1)))

        return 0.5 * torch.sum(regularizer ** 2) / (self.A.size(1) ** 2)

    def _sym_project(self, X):
        return torch.mm(self.symbolic_kernel(X), self.A)
