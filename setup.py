from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup_requires = [
    ]

install_requires = [
    'torch>=1.9.0',
    'gpytorch>=1.6.0',
    ]

dependency_links = [
    ]

setup(
    name='torch_ssge',
    version='0.1',
    author='Juhyeong Kim',
    author_email='nonconvexopt@gmail.com',
    python_requires='>=3.6',
    install_requires=install_requires,
    setup_requires=setup_requires,
    dependency_links=dependency_links,
    py_modules=['core'],
)