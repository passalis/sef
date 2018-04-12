# PySEF: A Python Library for Similarity-based Dimensionality Reduction
This package provides an implementation of the [Similarity Embedding Framework](https://arxiv.org/abs/1706.05692) (SEF) on top of the *PyTorch* library. The documentation of PySEF is available [here](https://pysef.readthedocs.io).

[![Build Status](https://travis-ci.org/passalis/sef.svg?branch=master)](https://travis-ci.org/passalis/sef)
[![Documentation Status](https://readthedocs.org/projects/pysef/badge/?version=latest)](http://pysef.readthedocs.io/en/latest/?badge=latest)

## What is the Similarity Embedding Framework?
The vast majority of Dimensionality Reduction techniques rely on second-order statistics to define their optimization objective. Even though this provides adequate results in most cases, it comes with several shortcomings. The methods require carefully designed regularizers and they are usually prone to outliers. The Similarity Embedding Framework can overcome the aforementioned limitations and provides a conceptually simpler way to express optimization targets similar to existing DR techniques. Deriving a new DR technique using the Similarity Embedding Framework becomes simply a matter of choosing an appropriate target similarity matrix. A variety of classical tasks, such as performing supervised dimensionality reduction and providing out-of-of-sample extensions, as well as, new novel techniques, such as providing fast linear embeddings for complex techniques, are demonstrated. 

## How to install PySEF?
You can simply *pip install pysef*! 

Be sure to check the [documentation](https://pysef.readthedocs.io) of the project.

## Further reading

You can find more details about the Similarity Embedding Framework in our [paper](https://arxiv.org/abs/1706.05692).

## Citation

If you find PySEF/SEF useful please cite the relevant publications. The PySEF library is described in:
<pre>
@article{passalis2018pysef,
  title={PySEF: A Python Library for Similarity-based Dimensionality Reduction},
  author={Passalis, Nikolaos and Tefas, Anastasios},
  journal={Knowledge-Based Systems},
  volume={(to appear)},
  pages={(to appear)},
  year={2018},
}
</pre>
while the method was derived and evaluated in detail in:

<pre>
@article{passalis2017sef,
  title={Dimensionality Reduction Using Similarity-Induced Embeddings},
  author={Passalis, Nikolaos and Tefas, Anastasios},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  volume={(to appear)},
  pages={(to appear)},
  year={2018},
}
</pre>
