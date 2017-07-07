from setuptools import setup
from sef_dr.version import __version__

setup(
    name='PySEF',
    version=__version__,
    author='N. Passalis',
    author_email='passalis@csd.auth.gr',
    packages=['sef_dr',],
    url='https://github.com/passalis/sef_dr',
    license='LICENSE.txt',
    description='Package that implements the Similatiry Embedding Framework on top of the theano.',
    install_requires=[
        "lasagne >= 0.2.dev1",
        "scikit-learn >= 0.18.1",
        "theano>= 0.9",
    ],
)
