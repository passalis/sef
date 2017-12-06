.. PySEF documentation master file, created by
   sphinx-quickstart on Wed Dec  6 00:29:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PySEF's documentation!
=================================

*PySEF* provides an implementation of the Similarity Embedding Framework (SEF) on top of the *PyTorch* library. *PySEF* an easy to use scikit-learn-like interface, allows for easily implementing novel dimensionality reduction techniques and can efficiently handle large-scale dataset using the GPU.


What is the Similarity Embedding Framework?
-------------------------------------------


The vast majority of Dimensionality Reduction techniques rely on second-order statistics to define their optimization objective. Even though this provides adequate results in most cases, it comes with several shortcomings. The methods require carefully designed regularizers and they are usually prone to outliers. The Similarity Embedding Framework can overcome the aforementioned limitations and provides a conceptually simpler way to express optimization targets similar to existing DR techniques. Deriving a new DR technique using the Similarity Embedding Framework becomes simply a matter of choosing an appropriate target similarity matrix. A variety of classical tasks, such as performing supervised dimensionality reduction and providing out-of-of-sample extensions, as well as, new novel techniques, such as providing fast linear embeddings for complex techniques, are demonstrated.

.. toctree::
   :maxdepth: 3
   :caption: Contents:
    
   installation
   usage
   extending


