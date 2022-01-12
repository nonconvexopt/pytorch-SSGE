import pytest


# Example is consist of (sampler instance, <log density method>)
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


def test_exponential_family():
    assert 