import torch
from pykeops.torch import LazyTensor

x, y = torch.randn(1000, 3), torch.randn(2000, 3)
x_i, y_j = LazyTensor(x[:, None, :]), LazyTensor(y[None, :, :])

K_keops = (
    -((x_i - y_j) ** 2).sum(2)
).exp()  # Symbolic (1000,2000) Gaussian kernel matrix

K_torch = (
    -((x[:, None, :] - y[None, :, :]) ** 2).sum(2)
).exp()  # Explicit (1000,2000) Gaussian kernel matrix

w = torch.rand(1000, 2)


def test_transpose():
    assert torch.allclose(K_keops.t() @ w, K_torch.t() @ w)
