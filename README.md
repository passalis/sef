# Similarity Embedding Framework
This package provides an implementation of the Similarity Embedding Framework (SEF) on top of the theano library.

## What is the Similarity Embedding Framework?
The vast majority of Dimensionality Reduction techniques rely on second-order statistics to define their optimization objective. Even though this provides adequate results in most cases, it comes with several shortcomings. The methods require carefully designed regularizers and they are usually prone to outliers. The Similarity Embedding Framework can overcome the aforementioned limitations and provides a conceptually simpler way to express optimization targets similar to existing DR techniques. Deriving a new DR technique using the Similarity Embedding Framework becomes simply a matter of choosing an appropriate target similarity matrix. A variety of classical tasks, such as performing supervised dimensionality reduction and providing out-of-of-sample extensions, as well as, new novel techniques, such as providing fast linear embeddings for complex techniques, are demonstrated in this paper using the proposed framework. 

## What SEF can do?

### 1. Recreate the geometry of a high dimensional space into a space with less dimensions 

In [examples/unsupervised_approximation.py](unsupervised_approximation.py) we recreate the 20-d PCA using just 10 dimensions:

| Method     | Accuracy |
| --------|---------|
| PCA 10-d  |  82.88%  | 
| SEF mimics PCA-20d | 84.59% | 


## How to install SEF?

## How to define custom target functions?



