*************
Using PySEF
*************

How to get started?
===================
After installing PySEF (see :ref:`installation-link`), simply import sef_dr, create a SEF object, fit it and transform your data::

    import sef_dr
    proj = sef_dr.LinearSEF(input_dimensionality=784, output_dimensionality=9)
    proj.fit(data=data, target_labels=data, target='supervised', iters=10)
    transformed_data = proj.transform(data)


The *input_dimensionality* parameter defines the dimensionality of the input data, while the *output_dimensionality* refers to the desired dimensionality of the data. Then, we can learn the projection using the *.fit()* function. The method that will be used for reducing the dimensionality of the data is specified in the *target* parameter of the *.fit()* method (PySEF provides many predefined targets/methods for dimensionality reduction, even though new methods can be also easily implemented as shown in :ref:`extending-link`). Several different dimensionality reduction scenarios are discussed in the following sections (for all the conducted experiments the well-known `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset is used).

Using GPU acceleration
======================

Following the PyTorch calling conventions, to use the GPU for the optimization/projection the *.cuda()* method can be used::

    proj.cuda()

To move the model back to cpu, the *.cpu()* method should be called::

    proj.cpu()


Recreating the geometry of a high dimensional space into a space with less dimensions
=====================================================================================
In `unsupervised_approximation.py <https://github.com/passalis/sef/blob/master/examples/unsupervised_approximation.py>`_ we demonstrate how to recreate the 50-d PCA using just 10 dimensions::

    # Learn a high dimensional projection
    proj_to_copy = PCA(n_components=50)
    proj_to_copy.fit(train_data[:n_train_samples, :])
    target_data = np.float32(proj_to_copy.transform(train_data[:n_train_samples, :]))

    # Approximate it using the SEF and 10 dimensions
    proj = LinearSEF(train_data.shape[1], output_dimensionality=10)
    proj.cuda()
    loss = proj.fit(data=train_data[:n_train_samples, :], target_data=target_data, target='copy', epochs=50, batch_size=128, verbose=True, learning_rate=0.001, regularizer_weight=0.001)

    # Evaluate the method
    acc = evaluate_svm(proj.transform(train_data[:n_train_samples, :]), train_labels[:n_train_samples], proj.transform(test_data), test_labels)

The experimental results demonstrate the ability of the proposed method to efficiently recreate the geometry of a high dimensional space into a space with less dimensions:

=============================   ==========
Method                          Accuracy
=============================   ==========
PCA 10-d                        82.88%
**Linear SEF mimics PCA-20d**   **84.87%**
=============================   ==========


Re-deriving similarity-based versions of well-known techniques
===============================================================
In `supervised_reduction.py <https://github.com/passalis/sef/blob/master/examples/supervised_reduction.py>`_ we demonstrate how to rederive similarity-based versions of well-known techniques. More specifically, a similarity-based LDA-like technique is derived::

    proj = LinearSEF(train_data.shape[1], output_dimensionality=(n_classes - 1))
    proj.cuda()
    loss = proj.fit(data=train_data[:n_train, :], target_labels=train_labels[:n_train], epochs=50, target='supervised', batch_size=128, regularizer_weight=0.001, verbose=True)


The SEF-based method leads to superior results:


==============    ==============   ==========
Method            Dimensionality   Accuracy
==============    ==============   ==========
LDA               9d	           85.66%
Linear SEF        9d	           88.89%
**Linear SEF**    **18d**          **89.48%**
==============    ==============   ==========


Providing out-of-sample extensions
===================================

In `linear_outofsample.py <https://github.com/passalis/sef/blob/master/examples/linear_outofsample.py>`_ and `kernel_outofsample.py <https://github.com/passalis/sef/blob/master/examples/kernel_outofsample.py>`_ we use the SEF to provide (linear and kernel) out-of-sample extensions for the ISOMAP technique. Note that the SEF, unlike the regression-based method, is not limited by the number of dimensions of the original technique::

    isomap = Isomap(n_components=10, n_neighbors=20)
    train_data_isomap = np.float32(isomap.fit_transform(train_data[:n_train_samples, :]))
    proj = LinearSEF(train_data.shape[1], output_dimensionality=10)
    proj.cuda()
    loss = proj.fit(data=train_data[:n_train_samples, :], target_data=train_data_isomap, target='copy', epochs=50, batch_size=128, verbose=True, learning_rate=0.001, regularizer_weight=0.001)

The results are shown in the following tables:

==================    ==============   ============
Method                Dimensionality   Accuracy
==================    ==============   ============
Linear Regression     10d              85.25%
Linear SEF            10d              85.76%
**Linear SEF**        **20d**            **89.48%**
==================    ==============   ============

==================    ==============   ==========
Method                Dimensionality   Accuracy
==================    ==============   ==========
Kernel Regression     10d              89.48%
Kernel SEF            10d              88.60%
**Kernel SEF**        **20d**          **90.88%**
==================    ==============   ==========


Performing SVM-based analysis
=============================

Finally, in `svm_approximation.py <https://github.com/passalis/sef/blob/master/examples/svm_approximation.py>`_  an SVM-based analysis technique that mimics the similarity induced by the hyperplanes of the 1-vs-1 SVMs is used to perform DR. This method allows for using a light-weight classifier, such as the NCC, to perform fast classification::

    # Learn an SVM
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    parameters = {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
    model = grid_search.GridSearchCV(svm.SVC(max_iter=10000, decision_function_shape='ovo'), parameters, n_jobs=-1, cv=3)
    model.fit(train_data[:n_train], train_labels[:n_train])

    # Learn a similarity embedding
    params = {'model': model, 'n_labels': np.unique(train_labels).shape[0], 'scaler': scaler}
    proj = LinearSEF(train_data.shape[1], output_dimensionality=dims)
    proj.cuda()
    loss = proj.fit(data=train_data[:n_train, :], target_data=train_data[:n_train, :], target_labels=train_labels[:n_train], target='svm', target_params=params, epochs=50, learning_rate=0.001, batch_size=128, verbose=True, regularizer_weight=0.001)

This code repeatedly calls the SVM to calculate the similarity matrix for the samples in each batch. If the whole similarity matrix can fit into the memory, we can speed up this process by using a precomputed similarity matrix as follows::

    from sef_dr.targets import generate_svm_similarity_matrix, sim_target_svm_precomputed
    
    # Precompute the similarity matrix
    Gt = generate_svm_similarity_matrix(train_data, train_labels, len(np.unique(train_labels)), model, scaler)
    params = {'Gt': Gt}
    
    proj = LinearSEF(train_data.shape[1], output_dimensionality=dims)
    proj.cuda()
    loss = proj.fit(data=train_data, target_data=train_data, target_labels=train_labels, target=sim_target_svm_precomputed, target_params=params, epochs=50, learning_rate=0.001, batch_size=128, verbose=True, regularizer_weight=0.001)

The results are shown in the following table:

======================   ==============   ==========
Method                   Dimensionality   Accuracy
======================   ==============   ==========
NCC - Original           784d             80.84%
NCC - Linear SEF         10d              86.50%
**NCC - Linear SEF**     **20d**          **86.67%**
======================   ==============   ==========


More examples
=============
More examples using six different datasets (15-Scene, Corel, MNIST, Yale, KTH, 20NG) are provided on `Github <https://github.com/passalis/sef/blob/master/examples>`_. To run these examples you have to download the extracted descriptors from `datasets <https://www.dropbox.com/sh/9qlt6b54v5jxial/AABccAu09ngHWPoj7kc9HOaXa?dl=0>`_ into the *data* folder. Note that slight differences from the original research `paper <https://arxiv.org/abs/1706.05692>`_ are due to some minor changes (batch-based optimization, faster estimation of the scaling factor, port to PyTorch).

PySEF tutorials
===============

The capabilities of PySEF are thoroughly demonstrated in two ipython tutorials that can be found in `tutorials <https://github.com/passalis/sef/blob/master/tutorials>`_.


