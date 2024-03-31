# Test for operation involving i and j placeholders

import math
import torch

M, N, D, DV = 7, 10, 1, 1

dtype = torch.float32
device_id = "cuda:0" if torch.cuda.is_available() else "cpu"

x = torch.randn(M, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.randn(N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(N, DV, device=device_id, dtype=dtype)
I = torch.arange(M, device=device_id, dtype=dtype)
J = torch.arange(N, device=device_id, dtype=dtype)

axis = 0

# Testing with Genred syntax

from pykeops.torch import Genred

fun1 = Genred(
    "Exp(i-j*SqDist(x,y))*b", [f"x=Vi({D})", f"y=Vj({D})", f"b=Vj({DV})"], axis=axis
)
fun2 = Genred(
    "Exp(i-j*SqDist(x,y))*b",
    [f"x=Vi({D})", f"y=Vj({D})", f"b=Vj({DV})", "i=Vi(1)", "j=Vj(1)"],
    axis=axis,
)
res1 = fun1(x, y, b)
res2 = fun2(x, y, b, I, J)

print(torch.norm(res1 - res2) / torch.norm(res2))


# Testing with LazyTensor syntax

from pykeops.torch import LazyTensor, i, j

xi = LazyTensor(x[:, None, :])
yj = LazyTensor(y[None, :, :])
bj = LazyTensor(b[None, :, :])

dij2 = (xi - yj).sum() ** 2
Kij = (i - j * dij2).exp()
res3 = (Kij * bj).sum_reduction(axis=axis)

print(torch.norm(res3 - res2) / torch.norm(res2))


# Testing with LazyTensor syntax - alternate style

from pykeops.torch import Vi, Vj, i, j

xi, yj, bj = Vi(x), Vj(y), Vj(b)

dij2 = (xi - yj).sum() ** 2
Kij = (i - j * dij2).exp()
res4 = (Kij * bj).sum_reduction(axis=axis)

print(torch.norm(res4 - res2) / torch.norm(res2))


# Testing with batch dimensions and broadcasting

A, B, C, M, N, D, DV = 3, 4, 5, 7, 7, 1, 1

x = torch.randn(A, B, 1, M, 1, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.randn(1, B, 1, 1, N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(1, B, C, 1, N, DV, device=device_id, dtype=dtype)
I = torch.arange(M, device=device_id, dtype=dtype).view(1, 1, 1, M, 1, 1)
J = torch.arange(N, device=device_id, dtype=dtype).view(1, 1, 1, 1, N, 1)

axis = 3

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

print(torch.norm(res1 - res2) / torch.norm(res1))
