import pytest

import torch

import gpytorch
from torch_ssge import SSGE

KERNELS = [
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()),
]

# Example is consist of (<sampler instance>)
# <sampler instance> should have '.sample' and 'log_prob' method.
EXAMPLES = [
    [
        torch.distributions.uniform.Uniform,
        [
            {"low":torch.tensor([0.0]), "high":torch.tensor([1.0])},
            {"low":torch.tensor([-1.0]), "high":torch.tensor([1.0])},
            {"low":torch.tensor([-10.0]), "high":torch.tensor([10.0])},
        ],
    ],
    [
        torch.distributions.laplace.Laplace,
        [
            {"loc":torch.tensor([0.0]), "scale":torch.tensor([1.0])},
            {"loc":torch.tensor([1.0]), "scale":torch.tensor([0.5])},
            {"loc":torch.tensor([-1.0]), "scale":torch.tensor([2])},
        ],
    ],
    [
        torch.distributions.normal.Normal,
        [
            {"loc":torch.tensor([0.0]), "scale":torch.tensor([1.0])},
            {"loc":torch.tensor([1.0]), "scale":torch.tensor([0.5])},
            {"loc":torch.tensor([-1.0]), "scale":torch.tensor([2])},
        ],
    ],
    [
        torch.distributions.poisson.Poisson,
        [
            {"rate":torch.tensor([1])},
            {"rate":torch.tensor([0.5])},
            {"rate":torch.tensor([4])},
        ]        
    ],
    [
        torch.distributions.studentT.StudentT,
        [
            {"df":torch.tensor([2.0])},
            {"df":torch.tensor([4.0]), "loc":torch.tensor([1.0]), "scale":torch.tensor([2.0])},
        ]
    ],
    [
        torch.distributions.gamma.Gamma,
        [
            {"concentration":torch.tensor([1.0]), "rate":torch.tensor([1.0])},
            {"concentration":torch.tensor([2.0]), "rate":torch.tensor([2.0])},
        ]
    ],
    [
        torch.distributions.gumbel.Gumbel,
        [
            {"loc":torch.tensor([1.0]), "scale":torch.tensor([2.0])},
        ]
    ],
    [
        torch.distributions.dirichlet.Dirichlet,
        [
            {"concentration":torch.tensor([0.5, 0.5])},
            {"concentration":torch.tensor([0.9, 0.1])},
        ]
    ],
    [
        torch.distributions.multivariate_normal.MultivariateNormal,
        [
            {"loc":torch.randn(10), "covariance_matrix":torch.eye(10)},
        ]
    ]
    [
        torch.distributions.multinomial.Multinomial,
        [
            {"probs":torch.tensor([ 1., 1., 1., 1.])},
        ]
    ]
]




@pytest.mark.parametrize("gpytorch_kernel", KERNELS)
@pytest.mark.parametrize("torch_dist", EXAMPLES)
def test_exponential_family(gpytorch_kernel, dist_example):
    
    estimator = SSGE(
        gpytorch_kernel,
        noise=1e-3
    )
    
    dist_module, params = dist_example

    for param in params:
        dist = dist_module(param)

        sample = dist.sample((100, 1))
        estimator.fit(sample)

        mean = dist.mean
        sqrt = dist.var.sqrt()

        test_points = torch.linspace(mean - 3 * sqrt, mean + 3 * sqrt, 500)
        test_points.requires_grad_()
        grad_estimate = estimator(test_points)
        grad_analytical = torch.autograd.grad(
            dist.log_prob(test_points),
            test_points
        )[0]
        assert torch.allclose(grad_estimate, grad_analytical, atol = 0.1, rtol = 0.)
