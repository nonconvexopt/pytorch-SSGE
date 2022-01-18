import math
import torch
import gpytorch


class SSGE(torch.nn.Module):
    def __init__(self, kernel: gpytorch.kernels.Kernel, eig_prop_threshold: float=0.99, noise=1e-8):
        super(SSGE, self).__init__()
        self.kernel = kernel
        self.noise = noise

        # Save cloned samples and its gram matrix
        self.sample = None
        self.gram = None

        # Parameters required to inference estimating gradients of arbitary samples.
        self.eig_prop_threshold = eig_prop_threshold
        self.K = None
        self.eigval = None
        self.eigvec = None
        self.beta = None


    def fit(self, x: torch.Tensor) -> None:
        new_sample = x.clone().requires_grad_(True)

        if self.sample is not None:
            self.sample = torch.cat([self.sample, new_sample], axis=0)
        else:
            self.sample = new_sample

        self.M = self.sample.shape[0]
        self.dim = self.sample.shape[1]

        self.gram = self.kernel(self.sample, self.sample.clone()).evaluate()
        if self.noise:
            self.gram = self.gram + self.noise * torch.eye(self.gram.shape[-1], dtype=self.gram.dtype, device=self.gram.device)
        self.grad_mean_sample_K = 0.5 * torch.autograd.grad(
            outputs = self.gram.sum(),
            inputs = self.sample,
            retain_graph = True,
        )[0]

        # TODO: find mechanism to find largest eigvals considering the proportions
        #       while not caculating all eigenvalues in torch.lobpcg.
        eigval, eigvec = torch.lobpcg(self.gram, self.dim, method="ortho")
        with torch.no_grad():
            eig_props = eigval.cumsum(-1) / eigval.sum(-1, keepdims=True)
            eig_props *= eig_props < self.eig_prop_threshold
            self.J = torch.argmax(eig_props, -1)
        self.eigval = eigval[:self.J]
        self.eigvec = eigvec[:, :self.J]
        assert (self.eigval > 0).all(), "Kernel matrix is not postive definite."

        input_tensor = self.sample.unsqueeze(-1).repeat(1, 1, self.J)
        eigfun_hat = math.sqrt(self.M) * torch.einsum(
            "jnm,mk,k->j",
            self.kernel(torch.einsum("ndj->jnd", input_tensor), self.sample).evaluate(),
            self.eigvec,
            self.eigval.reciprocal()
        )

        # beta should have size d x j
        self.beta = - self.eigvec * torch.autograd.grad(
            outputs = eigfun_hat,
            inputs = input_tensor,
            retain_graph = True,
        )[0].mean(0)

        beta = - torch.einsum(eigvec, self.grad_mean_sample_K) / eigval.unsqueeze(-1)
        # grads: [..., N, x_dim]
        grads = tf.matmul(eigen_ext, beta)

        Kxq = self.gram(x, samples, kernel_width)
        # Kxq = tf.Print(Kxq, [tf.shape(Kxq)], message="Kxq:")
        # ret: [..., N, n_eigen]
        ret = tf.sqrt(tf.to_float(self.M)) * tf.matmul(Kxq, eigen_vectors)
        ret *= 1. / tf.expand_dims(eigen_values, axis=-2)
        return ret


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.beta is not None, "Train samples should be fitted with `.fit()` before estimating gradients."
        _x = x.clone().requires_grad_(True)

        gram_wing = self.kernel(_x, self.sample).evaluate()
        eigfun_hat = math.sqrt(self.M) * torch.einsum("nm,mj->nj", K_wing, self.eigvec) / self.eigval
        gradfun_hat = torch.einsum("nj,mj->nm", eigfun_hat, self.beta)
        return torch.autograd.grad(
            outputs = gradfun_hat,
            inputs = _x,
        )[0]
