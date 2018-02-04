from setuptools import setup
from sef_dr.version import __version__

setup(
    name='PySEF',
    version=__version__,
    author='N. Passalis',
    author_email='passalis@csd.auth.gr',
    packages=['sef_dr',],
    url='https://github.com/passalis/sef',
    license='LICENSE.txt',
    description='Package that implements the Similarity Embedding Framework on top of the PyTorch library.',
    setup_requires=[
        "scikit-learn >= 0.19.1",
	    "numpy >= 1.13.3",
        "scipy >= 1.0.0",
        "torchvision >= 0.2.0"
    ],
    install_requires=[
        "scikit-learn >= 0.19.1",
        "numpy >= 1.13.3",
        "scipy >= 1.0.0",
        "torchvision >= 0.2.0"

    ],
)
