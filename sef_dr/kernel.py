# License: MIT License https://github.com/passalis/sef/blob/master/LICENSE.txt

import numpy as np
import theano
import theano.tensor as T
from sklearn.decomposition import KernelPCA
from sef_dr.similarity import sym_distance_matrix
from lasagne.updates import adam
from sef_dr.similarity import mean_data_distance
from sef_base import SEF_Base


class KernelSEF(SEF_Base):
    def __init__(self, data, input_dimensionality, output_dimensionality, learning_rate=0.0001, kernel_type='rbf',
                 degree=2, sigma=0, kernel_scaling=1, c=1, regularizer_weight=0.001, scaler=None):
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
        :param c: constant kernel param for linear and poly kernsl
        :param regularizer_weight: weight of the regularizer
        :param scaler: the sklearn-compatible scaler (or None) 
        """
        # Call base constructor
        SEF_Base.__init__(self, input_dimensionality, output_dimensionality, learning_rate, scaler=scaler)

        # Adjustable parameters
        self.kernel_type = kernel_type
        self.degree = degree
        self.sigma_kernel = theano.shared(name='sigma_kernel', value=np.float32(sigma))
        self.alpha = kernel_scaling
        self.c = c

        # If scaler is used, fit it!
        if self.scaler is None:
            data = np.float32(data)
        else:
            data = np.float32(self.scaler.fit_transform(data))

        # If the rbf kernel is used and no sigma is supplied, estimate it!
        sigma_kernel = np.float32(mean_data_distance(data))

        if sigma == 0 and self.kernel_type == 'rbf':
            self.sigma_kernel.set_value(sigma_kernel)

        # Use kPCA for initialization
        kpca = KernelPCA(kernel=self.kernel_type, n_components=self.output_dimensionality,
                         gamma=(1.0 / (sigma_kernel ** 2)), degree=self.degree, eigen_solver='dense')
        kpca.fit(data)
        A = kpca.alphas_
        # Scale the coefficients to have unit norm (avoid rescaling)
        A = A / np.sqrt(np.diag(np.dot(A.T, np.dot(np.dot(data, data.T), A))))

        # Model parameters
        self.X_kernel = theano.shared(value=np.float32(data), name='X_kernel')
        self.A = theano.shared(value=np.float32(A), name='A', borrow=True)

        regularizer = T.dot(self.A.T, T.dot(self.symbolic_kernel(self.X_kernel), self.A)) \
                      - T.eye(self.A.shape[1], self.A.shape[1])
        regularizer_loss = 0.5 * (T.sum(regularizer ** 2)) / (self.A.shape[1]) ** 2

        loss = (2 - regularizer_weight) * self._sym_loss(self.X, self.Gt, self.Gt_mask) \
               + regularizer_weight * regularizer_loss

        updates = adam(loss, [self.A], learning_rate=self.learning_rate)
        self.train_fn = theano.function(inputs=[self.X, self.Gt, self.Gt_mask], outputs=loss, updates=updates)
        self.project_fn = theano.function(inputs=[self.X], outputs=self._sym_project_data(self.X))

    def _initialize(self, data):
        """
        Initializes the kernel SEF model
        :param data: Data to be used for the initialization
        :return: 
        """

        if self.scaler is None:
            data = np.float32(data)
        else:
            data = np.float32(self.scaler.transform(data))

        # Estimate the sigma projection value (this usually makes the optimization "easier")
        sigma = np.float32(mean_data_distance(self.transform(data)))
        self.sigma_projection.set_value(sigma)

    def _sym_project_data(self, X):
        return T.dot(self.symbolic_kernel(X), self.A)

    def symbolic_kernel(self, X):
        if self.kernel_type == 'linear':
            K = self.alpha * T.dot(X, T.transpose(self.X_kernel)) + self.c
        elif self.kernel_type == 'poly':
            K = (self.alpha * T.dot(X, T.transpose(self.X_kernel)) + self.c) ** self.degree
        elif self.kernel_type == 'rbf':
            D = sym_distance_matrix(self.X_kernel, X) ** 2
            K = T.exp(-D / (self.sigma_kernel ** 2))
            K = T.transpose(K)
        else:
            raise Exception('Unknown kernel type: ', self.kernel_type)
        return K
