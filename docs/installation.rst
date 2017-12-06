.. _installation-link:

*************
Installation
*************

Recommended (pip-based) installation
====================================
A working installation of PyTorch is required before installing PySEF. To install PyTorch, please follow the instructions given in `PyTorch site <http://http://pytorch.org/>`_.

The recommended way to install PySEF is simply to use the *pip* package manager::

    pip install pysef

All the other required dependecies will be automatically donwloaded and installed.

Bleeding-edge installation
===========================

To install the latest version availabe at github, clone our repository and manually install the package::

    git clone https://github.com/passalis/sef
    cd sef
    python setup.py install --user


Running the examples/tutorials
==============================

*Keras* is needed to run the examples and the tutorials provided in our github repository (since keras provide an easy way to load the *MNIST* dataset). Therefore, before running the supplied examples/tutorials you have to install keras (along with an appropriate backend, e.g., the *Tensorflow* library)::

    pip install keras
    pip install tensorflow
    
Note that these are only needed for the specific examples and they are not mandatory for running the *PySEF* library.
