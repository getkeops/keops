import torch
from pykeops.torch import LazyTensor

x, y = torch.randn(1000, 3), torch.randn(2000, 3)
x_i, y_j = LazyTensor(x[:, None, :]), LazyTensor(y[None, :, :])
K = (-((x_i - y_j) ** 2).sum(2)).exp()  # Symbolic (1000,2000) Gaussian kernel matrix
K_ = (
    -((x[:, None, :] - y[None, :, :]) ** 2).sum(2)
).exp()  # Explicit (1000,2000) Gaussian kernel matrix
w = torch.rand(1000, 2)

print((K.t() @ w - K_.t() @ w).abs().mean())
