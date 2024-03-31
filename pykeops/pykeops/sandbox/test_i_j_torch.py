# Test for operation involving i and j placeholders

import math
import torch
from pykeops.torch import Genred, LazyTensor, i, j

dtype = torch.float32
device_id = "cuda:0" if torch.cuda.is_available() else "cpu"
backend = "GPU_2D" if torch.cuda.is_available() else "CPU"

M, N, D, DV = 1000, 1501, 1, 1

x = torch.randn(M, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.randn(N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(N, DV, device=device_id, dtype=dtype)
I = torch.arange(M, device=device_id, dtype=dtype)
J = torch.arange(N, device=device_id, dtype=dtype)

axis = 0

# Testing with Genred syntax

fun1 = Genred(
    f"Exp(-(i/{M})-(j/{N})*SqDist(x,y))*b", [f"x=Vi({D})", f"y=Vj({D})", f"b=Vj({DV})"], axis=axis,
)
fun2 = Genred(
    f"Exp(-(i/{M})-(j/{N})*SqDist(x,y))*b",
    [f"x=Vi({D})", f"y=Vj({D})", f"b=Vj({DV})", "i=Vi(1)", "j=Vj(1)"],
    axis=axis,
)
res1 = fun1(x, y, b, backend=backend)
res2 = fun2(x, y, b, I, J, backend=backend)

print(torch.norm(res1 - res2) / torch.norm(res2))

# Testing with LazyTensor syntax

xi = LazyTensor(x[:, None, :])
yj = LazyTensor(y[None, :, :])
bj = LazyTensor(b[None, :, :])

dij2 = (xi - yj).sum() ** 2
Kij = (-(i/M) - (j/N) * dij2).exp()
res3 = (Kij * bj).sum_reduction(axis=axis)

print(torch.norm(res3 - res2) / torch.norm(res2))


# Testing with LazyTensor syntax - alternate style

from pykeops.torch import Vi, Vj, i, j

xi, yj, bj = Vi(x), Vj(y), Vj(b)

dij2 = (xi - yj).sum() ** 2
Kij = (-(i/M) - (j/N) * dij2).exp()
res4 = (Kij * bj).sum_reduction(axis=axis)

print(torch.norm(res4 - res2) / torch.norm(res2))


# Testing with batch dimensions and broadcasting

A, B, C, M, N, D, DV = 3, 5, 7, 1501, 2000, 1, 1

x = torch.randn(A, B, 1, M, 1, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.randn(1, B, 1, 1, N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(1, B, C, 1, N, DV, device=device_id, dtype=dtype)
I = torch.arange(M, device=device_id, dtype=dtype).view(1, 1, 1, M, 1, 1)
J = torch.arange(N, device=device_id, dtype=dtype).view(1, 1, 1, 1, N, 1)

axis = 4

dij2 = (x - y).sum(axis=-1) ** 2
dij2 = dij2[...,None]
Kij = (-(I / M) - (J / N) * dij2).exp()
res0 = (Kij * b).sum(axis=axis)

xi = LazyTensor(x)
yj = LazyTensor(y)
bj = LazyTensor(b)
Ii = LazyTensor(I)
Jj = LazyTensor(J)

dij2 = (xi - yj).sum() ** 2
Kij = (-(Ii / M) - (Jj / N) * dij2).exp()
res1 = (Kij * bj).sum_reduction(axis=axis)

Kij = (-(i / M) - (j / N) * dij2).exp()
res2 = (Kij * bj).sum_reduction(axis=axis)

print(torch.norm(res1 - res0) / torch.norm(res0))
print(torch.norm(res2 - res0) / torch.norm(res0))
