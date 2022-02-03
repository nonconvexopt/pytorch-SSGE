import pytest

import torch

import gpytorch
from torch_ssge import SSGE

# Example is consist of (<sampler instance>)
# <sampler instance> should have '.sample' and 'log_prob' method.
EXAMPLES = [
    torch.distributions.uniform.Uniform,
    torch.distributions.laplace.Laplace,
    torch.distributions.normal.Normal,
    torch.distributions.poisson.Poisson,
    torch.distributions.studentT.StudentT,
    torch.distributions.gamma.Gamma,
    torch.distributions.gumbel.Gumbel,
    torch.distributions.dirichlet.Dirichlet,
    torch.distributions.multivariate_normal.MultivariateNormal,
    torch.distributions.multinomial.Multinomial,
]


@pytest.mark.parametrize("torch_dist", EXAMPLES)
def test_exponential_family(torch_dist):
    dist = torch_dist()

    estimator = SSGE(
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
        # gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()),
        noise=1e-3
    )

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
    assert torch.allclose(grad_estimate, grad_analytical)
