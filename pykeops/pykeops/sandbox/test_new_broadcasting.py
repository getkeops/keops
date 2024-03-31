# Test for operation involving i and j placeholders

import math
import torch

dtype = torch.float32
device_id = "cuda:0" if torch.cuda.is_available() else "cpu"
from pykeops.torch import LazyTensor

# Testing with batch dimensions and broadcasting

A, B, C, M, N, D, DV = 3, 4, 5, 7, 7, 1, 1

x = torch.randn(A, B, 1, M, 1, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.randn(1, B, 1, 1, N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(1, B, C, 1, N, DV, device=device_id, dtype=dtype)

axis = 3

dij2 = (x - y).sum(-1, keepdim=True) ** 2
Kij = (-dij2).exp()
res_torch = (Kij * b).sum(axis=axis)

xi = LazyTensor(x)
yj = LazyTensor(y)
bj = LazyTensor(b)

dij2 = (xi - yj).sum() ** 2
Kij = (-dij2).exp()
res_keops = (Kij * bj).sum_reduction(axis=axis)

print((torch.norm(res_keops - res_torch) / torch.norm(res_torch)).item())
