.. _examples-link:

***********************
Examples and Tutorials
***********************

We provide `examples <https://github.com/passalis/sef/blob/master/examples>`_ using six different datasets (15-Scene, Corel, MNIST, Yale, KTH, and 20NG) to reproduce the results obtained in the original research paper. Note that slight differences from the original `paper <https://arxiv.org/abs/1706.05692>`_ are due to some changes (batch-based optimization, faster estimation of the scaling factor, port to PyTorch). For all the reported results a `Linear SVM <https://github.com/passalis/sef/blob/master/sef_dr/classification.py>`_ is trained, unless otherwise stated.


Prerequisites
=============
To run the examples you have to install PySEF (please also refer to :ref:`installation-link`)::

     pip install pysef

Before running any of the following examples, please download the pre-extracted feature vectors from the following `drobpox folder <https://www.dropbox.com/sh/9qlt6b54v5jxial/AABccAu09ngHWPoj7kc9HOaXa?dl=0>`_  into a folder named *data* and execute the script from the same root folder (or simply update the data path in the code). Refer to :ref:`Data loading <data-loading>` for more details.

Please also install *matplotlib*, which is also needed for some of the following examples/tutorials::
    
    pip install matplotlib


Linear approximation of a high-dimensional technique
====================================================

In `unsupervised_approximation_multiple.py <https://github.com/passalis/sef/blob/master/examples/unsupervised_approximation_multiple.py>`_ we demonstrate how to recreate the 50-d PCA using just 10 dimensions. The proposed method (abbreviated as S-PCA) is compared to the 10-d PCA method. To run the example, simply download the aforementioned file and execute it::

    python unsupervised_approximation_multiple.py

The following results should be obtained:

=========== ========  ==========
Dataset	    PCA       S-PCA      
=========== ========  ==========
15-scene    61.94%    **67.20%**
Corel       36.18%    **38.55%**
MNIST       82.88%    **84.71%**
Yale        56.69%    **65.16%**
KTH         76.82%    **86.56%**
20NG        39.73%    **45.79%**
=========== ========  ==========



Supervised dimensionality reduction
===================================

In `supervised_reduction_multiple.py <https://github.com/passalis/sef/blob/master/examples/supervised_reduction_multiple.py>`_ we demonstrate how to perform supervised dimensionality reduction using the SEF. Two different setups are used: a) *S-LDA-1*, where the same dimensionality as the LDA method is used, and b) *S-LDA-2*, where the number of dimensions is doubled. To run the example, simply download the aforementioned file and execute it::

    python supervised_reduction_multiple.py

The following results should be obtained:

=========== ========  =============   ==============
Dataset	    LDA       S-LDA-1         S-LDA-2
=========== ========  =============   ==============
15-scene    66.76%    75.58%          **76.98%**
Corel       37.28%    **42.58%**      42.33%
MNIST       85.66%    89.03%          **89.27%**
Yale        93.95%    92.50%          **92.74%**
KTH         90.38%    90.73%          **91.66%**
20NG        63.57%    **70.35%**      70.25%
=========== ========  =============   ==============



Providing out-of-sample-extensions
===================================

In `linear_outofsample_mutiple.py <https://github.com/passalis/sef/blob/master/examples/linear_outofsample_mutiple.py>`_ we demonstrate how to provide out-of-sample extensions for the ISOMAP technique. Two different setups are used: a) *cS-ISOMAP-1*, where the dimensionality of the projection is set to 10, and b) *cS-ISOMAP-2*, where the dimensionality of the projection is set to 20. The proposed method is compared to performing linear regression (LR). To run the example, simply download the aforementioned file and execute it::

    python linear_outofsample_mutiple.py

The following results should be obtained:

=========== ========  =============   ==============
Dataset	    LR        cS-ISOMAP-1     cS-ISOMAP-2  
=========== ========  =============   ==============
15-scene    58.29%     67.26%         **69.04%**
Corel       34.93%     38.70%         **40.45%**
MNIST       85.11%     85.93%         **93.37%**
Yale        35.97%     62.09%         **82.58%**
KTH         67.20%     86.56%         **89.80%**
20NG        33.14%     41.52%         **47.97%**
=========== ========  =============   ==============

Kernel extensions can be also used (`kernel_outofsample_mutiple.py <https://github.com/passalis/sef/blob/master/examples/kernel_outofsample_mutiple.py>`_). The following results should be obtained:


=========== ========  =============   ==============
Dataset	    KR        cKS-ISOMAP-1    cKS-ISOMAP-2  
=========== ========  =============   ==============
15-scene    60.10%    63.89%          **68.14%**
Corel       36.22%    37.85%          **42.27%**
MNIST       89.48%    88.30%          **91.35%**
Yale        46.94%    29.84%          **62.25%**
KTH         72.31%    78.22%          **83.31%**
20NG        44.50%    41.57%          **48.81%**
=========== ========  =============   ==============


SVM-based analysis
==================
PySEF can be used to mimic the similarity induced by the hyperplanes of the 1-vs-1 SVMs and perform DR (`svm_approximation_multiple.py <https://github.com/passalis/sef/blob/master/examples/svm_approximation_multiple.py>`_). The proposed technique is combined with a lightweight Nearest Centroid Classifier. Two different setups are used: a) *S-SVM-A-1*, where the dimensionality of the projection is set to the number of classes, and b) *S-SVM-A-1*, where the dimensionality of the projection is set to twice the number of classes. To run the example, simply download the aforementioned file and execute it::

    python svm_approximation_multiple.py 

The following results should be obtained:

=========== ========  =============   ==============
Dataset	    Original  S-SVM-A-1       S-SVM-A-1
=========== ========  =============   ==============
15-scene    59.67%     **74.47%**       74.10%
Corel       37.40%     **42.15%**       41.77%
MNIST       80.84%     86.71%           **86.80%**
Yale        13.95%     84.44%           **88.63%**
KTH         79.72%     92.24%           **94.09%**
20NG        60.79%     65.37%           **65.78%**
=========== ========  =============   ==============



PySEF tutorials
===============

To run the tutorials you have to install the Jupyter Notebook (also refer to `Installing Jupyter <http://jupyter.readthedocs.io/en/latest/install.html>`_)::

    pip install jupyter

Then, download the notebook tutorial you are interested in. Currently two tutorial are available: a) `Supervised dimensionality reduction <https://github.com/passalis/sef/blob/master/tutorials/Supervised%20DR.ipynb>`_, and b) `Defining new dimensionality reduction methods <https://github.com/passalis/sef/blob/master/tutorials/Defining%20new%20methods.ipynb>`_. Then, navigate to the folder that contains the notebook and start the Jupyter Notebook::

    jupyter notebook

Finally, navigate to the default URL of Jupyter web app (`http://localhost:8888 <http://localhost:8888>`_) and select the notebook. Please make sure that you appropriately update the folder that contains the MNIST dataset when running the tutorials (refer to :ref:`Data loading <data-loading>` for more details, or just create an empty folder named *data* in the same root folder as the notebook and the dataset will be automatically downloaded).


