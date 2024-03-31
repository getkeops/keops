# Test for operation involving i and j placeholders

import math
import numpy as np

M, N, D, DV = 7, 10, 1, 1

dtype = np.float32

x = np.random.randn(M, D).astype(dtype) / math.sqrt(D)
y = np.random.randn(N, D).astype(dtype) / math.sqrt(D)
b = np.random.randn(N, DV).astype(dtype)
i = np.arange(M).astype(dtype)
j = np.arange(N).astype(dtype)

axis = 1

# Testing with Genred syntax

from pykeops.numpy import Genred

fun1 = Genred(
    "Exp(i-j*SqDist(x,y))*b", [f"x=Vi({D})", f"y=Vj({D})", f"b=Vj({DV})"], axis=axis
)
fun2 = Genred(
    "Exp(i-j*SqDist(x,y))*b",
    [f"x=Vi({D})", f"y=Vj({D})", f"b=Vj({DV})", "i=Vi(1)", "j=Vj(1)"],
    axis=axis,
)
res1 = fun1(x, y, b)
res2 = fun2(x, y, b, i, j)

print(np.linalg.norm(res1 - res2) / np.linalg.norm(res2))


# Testing with LazyTensor syntax

from pykeops.numpy import Vi, Vj, i, j

xi, yj, bj = Vi(x), Vj(y), Vj(b)

dij2 = (xi - yj).sum() ** 2
Kij = (i - j * dij2).exp()
res3 = (Kij * bj).sum_reduction(axis=axis)

print(np.linalg.norm(res3 - res2) / np.linalg.norm(res2))
