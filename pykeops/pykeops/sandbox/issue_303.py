import torch
from pykeops.torch import LazyTensor


# Details of this function don't matter
def fn(x_i, x_j, y_j):
    x_i = LazyTensor(x_i)
    x_j = LazyTensor(x_j)
    y_j = LazyTensor(y_j)
    K_ij = (-((x_i - x_j) ** 2).sum(-1)).exp()
    return (K_ij * y_j).sum(1)


# Batching with KeOps: runs
x_i = torch.randn(5, 10, 1, 2)
x_j = torch.randn(5, 1, 20, 2)
y_j = torch.randn(5, 1, 20, 1)
fn(x_i, x_j, y_j)

# Batching with vmap: raises exception
torch.vmap(fn)(x_i, x_j, y_j)
