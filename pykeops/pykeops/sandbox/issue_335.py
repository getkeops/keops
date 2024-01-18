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


if __name__ == "__main__":
    device = "cuda:0"
    train_x = torch.randn(30, 10000, 3, device=device)

    # covar_module = gpytorch.kernels.keops.MaternKernel(nu=2.5).to(device)
    # mat = covar_module(train_x)

    mat = covar_func(train_x)

    print(mat.shape)

    eye = torch.eye(10000, device=device)
    mat @ eye
