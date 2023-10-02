import torch
from pykeops.torch import LazyTensor


def fn_torch(x_i, x_j, y_j):
    K_ij = (-((x_i - x_j) ** 2).sum(-1)).exp()
    K_ij = K_ij[..., None]
    return ((K_ij * y_j).sum(1))


def fn_keops(x_i, x_j, y_j):
    x_i = LazyTensor(x_i)
    x_j = LazyTensor(x_j)
    y_j = LazyTensor(y_j)
    K_ij = (-((x_i - x_j) ** 2).sum(-1)).exp()
    return ((K_ij * y_j).sum(1))


x_i = torch.randn(10, 1, 2)
x_j = torch.randn(1, 20, 2)
y_j = torch.randn(1, 20, 1)

# testing torch.func.jvp.
_, res1 = torch.func.jvp(fn_torch, (x_i, x_j, y_j), (x_i, x_j, y_j))
print("res1=", res1)

_, res2 = torch.func.jvp(fn_keops, (x_i, x_j, y_j), (x_i, x_j, y_j))
print("testing jvp, error=",torch.norm(res1 - res2) / torch.norm(res1))
