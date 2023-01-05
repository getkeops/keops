import pykeops

pykeops.clean_pykeops()

import torch
from pykeops.torch import LazyTensor

H = 100
N = 1000
L = 5000
D = 10

dtype = torch.complex64
# dtype = torch.float32

# Complex
x_i = LazyTensor(torch.randn(H, N, 1, 1, dtype=dtype, requires_grad=True))
y_j = LazyTensor(torch.randn(1, 1, L, D, dtype=dtype, requires_grad=True))

D_ij = x_i * y_j

a_i = D_ij.sum(dim=1)

a_i.sum().backward()
