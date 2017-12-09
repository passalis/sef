from sef_dr.kernel import KernelSEF
import numpy as np
from sklearn.neighbors import  NearestCentroid


def test_kernel_sef():
    """
    Performs some basic testing using the KernelSEF
    :return:
    """
    np.random.seed(1)
    train_data = np.random.randn(100, 50)
    train_labels = np.random.randint(0, 2, 100)

    proj = KernelSEF(train_data, 50, output_dimensionality=12, kernel_type='rbf')
    proj._initialize(train_data)
    proj_data = proj.transform(train_data, batch_size=8)
    assert proj_data.shape[0] == 100
    assert proj_data.shape[1] == 12

    ncc = NearestCentroid()
    ncc.fit(proj_data, train_labels)
    acc_before = ncc.score(proj_data, train_labels)

    loss = proj.fit(data=train_data, target_labels=train_labels, epochs=200,
                    target='supervised', batch_size=8, regularizer_weight=0, learning_rate=0.0001,  verbose=False)

    # Ensure that loss is reducing
    assert loss[0] > loss[-1]

    proj_data = proj.transform(train_data, batch_size=8)
    assert proj_data.shape[0] == 100
    assert proj_data.shape[1] == 12

    ncc = NearestCentroid()
    ncc.fit(proj_data, train_labels)
    acc_after = ncc.score(proj_data, train_labels)

    assert acc_after > acc_before