import numpy as np
import theano
import theano.tensor as T
from sklearn.decomposition import PCA
from sef_base import SEF_Base
from sef_dr.similarity import mean_data_distance
from lasagne.updates import adam


class LinearSEF(SEF_Base):
    def __init__(self, input_dimensionality, output_dimensionality, learning_rate=0.001, regularizer_weight=0,
                 scaler=None):

        # Call base constructor
        SEF_Base.__init__(self, input_dimensionality, output_dimensionality, learning_rate, scaler)

        # Projection weights variables
        W = np.float32(0.1 * np.random.randn(self.input_dimensionality, output_dimensionality))
        self.W = theano.shared(value=W, name='W', borrow=True)

        # Define the SEF regularizer
        regularizer = T.dot(self.W.T, self.W) - T.eye(self.W.shape[1], self.W.shape[1])
        regularizer_loss = 0.5 * T.sum(regularizer ** 2) / (self.W.shape[1]) ** 2

        loss = (2 - regularizer_weight) * self._sym_loss(self.X, self.Gt, self.Gt_mask) \
               + regularizer_weight * regularizer_loss

        updates = adam(loss, [self.W], learning_rate=self.learning_rate)
        self.train_fn = theano.function(inputs=[self.X, self.Gt, self.Gt_mask], outputs=loss, updates=updates)
        self.project_fn = theano.function(inputs=[self.X], outputs=self._sym_project_data(self.X))

    def init(self, data):
        """
        Initializes the linear SEF model
        :param data: 
        :return: 
        """

        # Initialize values
        if self.scaler is None:
            data = np.float32(data)
        else:
            data = np.float32(self.scaler.fit_transform(data))

        # Use pca for initialization
        pca = PCA(n_components=self.output_dimensionality)
        pca.fit_transform(data)
        self.W.set_value(np.float32(pca.components_.transpose()))

        # Estimate the sigma projection value (this usually makes the optimization "easier")
        # This can be pre-set to scale the projection into a specific range
        sigma = mean_data_distance(self.transform(data))
        self.sigma_projection.set_value(np.float32(sigma))

    def _sym_project_data(self, X):
        return T.dot(X, self.W)
