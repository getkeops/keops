import torch
from pykeops.torch import LazyTensor

use_keops = True


# Details of this function don't matter
def fn(x_i, x_j, y_j, nbatchdims=0):
    if use_keops:
        x_i = LazyTensor(x_i)
        x_j = LazyTensor(x_j)
        y_j = LazyTensor(y_j)
    K_ij = (-((x_i - x_j) ** 2).sum(-1)).exp()
    if not use_keops:
        K_ij = K_ij[..., None]
    print((K_ij * y_j).shape)
    return (K_ij * y_j).sum(nbatchdims + 1)


# Batching with KeOps: runs
x_i = torch.randn(5, 10, 1, 2)
x_j = torch.randn(5, 1, 20, 2)
y_j = torch.randn(5, 1, 20, 1)
res1 = fn(x_i, x_j, y_j, nbatchdims=1)
print(res1.shape)

# Batching with vmap
res2 = torch.vmap(fn)(x_i, x_j, y_j)
print(res2.shape)

print(torch.norm(res1 - res2) / torch.norm(res1))
