import math

import torch

# import gpytorch

from pykeops.torch import LazyTensor as KEOLazyTensor


def covar_func(x1, x2=None):
    if x2 is None:
        x2 = x1

    x1_ = KEOLazyTensor(x1[..., :, None, :])
    x2_ = KEOLazyTensor(x2[..., None, :, :])

    distance = ((x1_ - x2_) ** 2).sum(-1).sqrt()
    exp_component = (-math.sqrt(5) * distance).exp()

    constant_component = (math.sqrt(5) * distance) + (1 + 5.0 / 3.0 * distance**2)

    return constant_component * exp_component


B, M, N, D = 25, 10000, 10000, 3

if __name__ == "__main__":
    device = "cuda:0"
    train_x = torch.randn(B, M, D, device=device)

    # covar_module = gpytorch.kernels.keops.MaternKernel(nu=2.5).to(device)
    # mat = covar_module(train_x)

    # mat = covar_func(train_x)
    rh = torch.rand(M, N, device=device)
    # res1 = mat @ rh

    mat = covar_func(train_x.view(B, 1, M, D))
    rh = rh.T.contiguous().view(1, N, 1, M, 1)
    res2 = (mat * rh).sum(axis=3).transpose(1, 2).view(B, M, N)

    # print("relative error:", (torch.norm(res1 - res2) / torch.norm(res1)).item())
