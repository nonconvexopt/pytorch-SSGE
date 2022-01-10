from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup


install_requires = [
    'torch>=1.9.0',
    'gpytorch>=1.6.0',
    ]

setup(
    name='torch-ssge',
    description='PyTorch implementation of the Spectral Stein Gradient Estimator.',
    version='0.1.0',
    author='Juhyeong Kim',
    author_email='nonconvexopt@gmail.com',
    url='git@github.com:nonconvexopt/pytorch_ssge.git',
    python_requires='>=3.6',
    install_requires=install_requires,
    packages=find_packages(exclude = []),
    py_modules=['core'],
)