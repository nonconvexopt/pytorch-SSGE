import pytest

import torch


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

@pytest.mark.parametrize("sampler", EXAMPLES)
def test_exponential_family(sampler):
    sampler.sample()