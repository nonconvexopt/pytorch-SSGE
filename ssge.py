import math
import torch
import gpytorch


class SSGE(torch.nn.Module):
    def __init__(self, kernel: gpytorch.kernels.Kernel, eig_prop_threshold: float = 0.99, noise = 1e-8):
        super(SSGE, self)__init__()
        self.kernel = kernel
        assert self.noise >= 0, 
        self.noise = noise
        self.eig_prop_threshold = eig_prop_threshold

        # Save samples directly
        self.sample = None
        self.gram = None

        # Parameters required to inference estimating gradients of arbitary samples.
        self.K = None
        self.eigval = None
        self.eigvec = None
        self.beta = None

    def fit(self, x: torch.Tensor) -> None:
        assert x.requires_grad, "'requires_grad' of input tensor must be set to True."

        if self.sample is not None:
            self.sample = torch.cat([self.sample, x], axis=0)
        else:
            self.sample = x

        m = self.sample.shape[0]
        self.dim = self.sample.shape[1]
        self.K = self.kernel(self.sample).evaluate()
        if self.noise:
            self.K = self.K + self.noise * torch.eye(m, dtype=self.sample.dtype, device=self.sample.device)

        eigval, eigvec = torch.linalg.eigh(self.K)
        # Tried to use torch.lobpcg as alternative to torch.linalg.eigh
        # but it is discovered that it is not stable yet.
        # self.eigval, self.eigvec = torch.lobpcg(self.K, min(int(m / 3), self.dim))
        with torch.no_grad():
            eig_props = eigval.cumsum(-1) / eigval.sum(-1, keepdims=True)
            eig_props *= eig_props < self.eig_prop_threshold
            self.j = torch.argmax(eig_props, -1)
        self.eigval = eigval[:self.j]
        self.eigvec = eigvec[:, :self.j]
        assert (self.eigval > 0).all(), "Kernel matrix is not postive definite."

        input_tensor = self.sample.unsqueeze(-1).repeat(1, 1, self.j)
        eigfun_hat = math.sqrt(m) * torch.einsum(
            "jnm,mk,k->j",
            self.kernel(torch.einsum("ndj->jnd", input_tensor), self.sample).evaluate(),
            self.eigvec,
            self.eigval.reciprocal()
        )
        # beta should have size d x j
        self.beta = - torch.autograd.grad(
            outputs = eigfun_hat,
            grad_outputs = torch.ones(eigfun_hat.shape, dtype=eigfun_hat.dtype, device=eigfun_hat.device),
            inputs = input_tensor,
            retain_graph = True,
        )[0].mean(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.beta is not None, "Train samples should be fitted with `.fit()` before estimating gradients."
        assert x.requires_grad, "'.requires_grad' of input tensor must be set to True."

        m = self.sample.shape[0]

        K_wing = self.kernel(x, self.sample).evaluate()
        eigfun_hat = math.sqrt(m) * torch.einsum("nm,mj->nj", K_wing, self.eigvec) / self.eigval
        gradfun_hat = torch.einsum("nj,mj->nm", eigfun_hat, self.beta)
        return torch.autograd.grad(
            outputs = gradfun_hat,
            grad_outputs = torch.ones(gradfun_hat.shape, dtype=gradfun_hat.dtype, device=gradfun_hat.device),
            inputs = x,
        )[0]
