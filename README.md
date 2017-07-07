# Similarity Embedding Framework
This package provides an implementation of the Similarity Embedding Framework (SEF) on top of the theano library.

## What is the Similarity Embedding Framework?
The vast majority of Dimensionality Reduction techniques rely on second-order statistics to define their optimization objective. Even though this provides adequate results in most cases, it comes with several shortcomings. The methods require carefully designed regularizers and they are usually prone to outliers. The Similarity Embedding Framework can overcome the aforementioned limitations and provides a conceptually simpler way to express optimization targets similar to existing DR techniques. Deriving a new DR technique using the Similarity Embedding Framework becomes simply a matter of choosing an appropriate target similarity matrix. A variety of classical tasks, such as performing supervised dimensionality reduction and providing out-of-of-sample extensions, as well as, new novel techniques, such as providing fast linear embeddings for complex techniques, are demonstrated in this paper using the proposed framework. 

## What SEF can do?

### 1. Recreate the geometry of a high dimensional space into a space with less dimensions!

In [examples/unsupervised_approximation.py](unsupervised_approximation.py) we recreate the 20-d PCA using just 10 dimensions:

| Method     | Accuracy |
| --------|---------|
| PCA 10-d  |  82.88%  | 
| Linear SEF mimics PCA-20d | **84.59%** | 

### 2. Re-derive similarity-based versions of well-known techniques!
In [examples/supervised_reduction.py.py](supervised_reduction.py) we derive a similarity-based LDA:


| Method     | Accuracy |
| --------|---------|
| LDA 9-d  |  85.67%  | 
| Linear SEF 9d | 88.24% | 
| Linear SEF 18d | **88.64%** | 

Note that LDA is limited to 9-d projections for classifications problems with 10 classes.

### 3. Provide out-of-sample extensions!
In [examples/linear_outofsample.py](linear_outofsample.py) and [examples/kernel_outofsample.py](kernel_outofsample.py) we use the SEF to provide out-of-sample extenstions for the ISOMAP technique. Note that the SEF, unlike the regression-based method, is not limited by the number of dimensions of the original technique.


| Method     | Accuracy |
| --------|---------|
| Linear Regression |  85.26%  | 
| Linear SEF 10d | 85.99% | 
| Linear SEF 20d |  **88.58%** | 

| Method     | Accuracy |
| --------|---------|
| Linear Regression |  89.48%  | 
| Kernel SEF 10d | 86.84 % | 
| Kernel SEF 20d | **90.43%** | 

### 4. Perform SVM-based analysis!
In [examples/svm_approximation.py](svm_approximation.py) an SVM-based analysis technique that mimics the similarity induced by the hyperplanes of the 1-vs-1 SVMs is used to perform DR. This method allows for using a light-weight classifier, such as the NCC, to perform fast classification.

| Method     | Accuracy |
| --------|---------|
| NCC - Original |  80.84%  | 
| NCC - Linear SEF 10d | 85.50 % | 
| NCC - Linear SEF 10d  | **86.41%** | 

(The MNIST dataset was used for all the conducted experiments)


## How to define custom target functions?

## How to install SEF?



