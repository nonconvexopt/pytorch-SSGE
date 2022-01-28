from setuptools import setup


install_requires = [
    'torch>=1.9.0',
    'gpytorch>=1.6.0',
    ]

setup(
    name='torch-ssge',
    description='Spectral Stein Gradient Estimator implemented  PyTorch',
    version='0.2.0',
    author='Juhyeong Kim',
    author_email='nonconvexopt@gmail.com',
    url='git@github.com:nonconvexopt/pytorch_ssge.git',
    python_requires='>=3.6',
    install_requires=install_requires,
    packages=['torch_ssge'],
    package_dir={'torch_ssge': 'torch_ssge'},
)
