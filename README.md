# Similarity Embedding Framework
This package provides an implementation of the Similarity Embedding Framework (SEF) on top of the *theano* library that follows the *scikit-learn* call conventions.

## What is the Similarity Embedding Framework?
The vast majority of Dimensionality Reduction techniques rely on second-order statistics to define their optimization objective. Even though this provides adequate results in most cases, it comes with several shortcomings. The methods require carefully designed regularizers and they are usually prone to outliers. The Similarity Embedding Framework can overcome the aforementioned limitations and provides a conceptually simpler way to express optimization targets similar to existing DR techniques. Deriving a new DR technique using the Similarity Embedding Framework becomes simply a matter of choosing an appropriate target similarity matrix. A variety of classical tasks, such as performing supervised dimensionality reduction and providing out-of-of-sample extensions, as well as, new novel techniques, such as providing fast linear embeddings for complex techniques, are demonstrated in this paper using the proposed framework. 

## What SEF can do?

### 1. Recreate the geometry of a high dimensional space into a space with less dimensions!

In [examples/unsupervised_approximation.py](examples/unsupervised_approximation.py) we recreate the 20-d PCA using just 10 dimensions:

| Method     | Accuracy |
| --------|---------|
| PCA 10-d  |  82.88%  | 
| Linear SEF mimics PCA-20d | **84.59%** | 

### 2. Re-derive similarity-based versions of well-known techniques!
In [examples/supervised_reduction.py.py](examples/supervised_reduction.py) we derive a similarity-based LDA:


| Method     | Accuracy |
| --------|---------|
| LDA 9d  |  85.67%  | 
| Linear SEF 9d | 88.24% | 
| Linear SEF 18d | **88.64%** | 

Note that LDA is limited to 9-d projections for classifications problems with 10 classes.

### 3. Provide out-of-sample extensions!
In [examples/linear_outofsample.py](examples/linear_outofsample.py) and [examples/kernel_outofsample.py](examples/kernel_outofsample.py) we use the SEF to provide out-of-sample extenstions for the ISOMAP technique. Note that the SEF, unlike the regression-based method, is not limited by the number of dimensions of the original technique.


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
In [examples/svm_approximation.py](examples/svm_approximation.py) an SVM-based analysis technique that mimics the similarity induced by the hyperplanes of the 1-vs-1 SVMs is used to perform DR. This method allows for using a light-weight classifier, such as the NCC, to perform fast classification.

| Method     | Accuracy |
| --------|---------|
| NCC - Original |  80.84%  | 
| NCC - Linear SEF 10d | 85.50 % | 
| NCC - Linear SEF 10d  | **86.41%** | 

### 5. Deriving a new DR method becomes simply a matter of defining a custom similarity target!

The SEF allows for easily deriving novel DR techniques by simply defining the target similarity matrix. For the current implementation this can be done by defining a function that follows this pattern:

```python
def my_custom_similarity_function(target_data, target_labels, sigma, idx, target_params):
    pass
```

The *target_data*, *target_labels*, *sigma*, and *target_params* are passed to the *.fit()* function. During the training this function is called with a different set of indices *idx* and it is expected to return the target similarity matrix for the data that corresponds to *idx*. For example, let's define a function that sets a target similarity of 0.8 for the points that belong to the same class, and 0 for the points that belong to different classes:


```python
def sim_target_supervised(target_data, target_labels, sigma, idx, target_params):

    cur_labels = target_labels[idx]
    N = cur_labels.shape[0]

    N_labels = len(np.unique(cur_labels))

    Gt, mask = np.zeros((N, N)), np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if cur_labels[i] == cur_labels[j]:
                Gt[i, j] = 0.8
                mask[i, j] = 1
            else:
                Gt[i, j] = 0
                mask[i, j] = 1.0 / (N_labels - 1)

    return np.float32(Gt), np.float32(mask)
```

Note that we also appropriately set the weighting mask to account for the imbalance between the intra-class and inter-class samples. It is important to remember to work only with the current batch (using the *idx*) and not the whole training set (that is always passed to *target_data*/*target_labels*). You can find more target function examples in [sef_dr/targets.py](sef_dr/targets.py) 


(The MNIST dataset was used for all the conducted experiments)


## How to install SEF?

TODO!

## Further reading

You can find more details about SEF in our [paper](https://arxiv.org/abs/1706.05692).

If you use SEF in your research, please cite the following paper:

<pre>
@article{passalis2017dimensionality,
  title={Dimensionality Reduction using Similarity-induced Embeddings},
  author={Passalis, Nikolaos and Tefas, Anastasios},
  journal={arXiv preprint arXiv:1706.05692},
  year={2017}
}
</pre>

