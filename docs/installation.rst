.. _installation-link:

*************
Installation
*************

Recommended (pip-based) installation
====================================
A working installation of PyTorch is required before installing PySEF. To install PyTorch, please follow the instructions given in the `PyTorch site <http://pytorch.org/>`_.

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



