.. _installation-link:

*************
Installation
*************

Recommended (pip-based) installation
====================================
A working installation of PyTorch is required before installing PySEF. To install PyTorch, please follow the instructions given in the `PyTorch site <http://http://pytorch.org/>`_.

The recommended way to install PySEF is simply to use the *pip* package manager::

    pip install pysef

All the other required dependecies will be automatically downloaded and installed.


PySEF is developed and tested on Linux (both Python 2.7 and Python 3.5 are supported). However, it is expected to run on Windows/Mac OSX as well, since all of its components are cross-platform.


Bleeding-edge installation
===========================

To install the latest version availabe at github, clone our repository and manually install the package::

    git clone https://github.com/passalis/sef
    cd sef
    python setup.py install --user


Running the examples/tutorials
==============================

*Keras* and *matplotlib* are also needed to run the examples and the tutorials provided in our github repository (keras provide an easy way to load the *MNIST* dataset and *matplotlib* to plot the loss function during the optimization). Therefore, before running the supplied examples/tutorials you have to install keras (along with an appropriate backend, e.g., the *Tensorflow* library) and matplotlib::

    pip install keras
    pip install tensorflow
    pip install matplotlib
    
Note that these are only needed for the specific examples and they are not mandatory for using the *PySEF* library.
