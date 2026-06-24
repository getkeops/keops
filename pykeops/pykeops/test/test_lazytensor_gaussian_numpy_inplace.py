import math
import numpy as np
from pykeops.numpy import LazyTensor
from pykeops.test import assert_np_allclose

M, N, D, DV = 3000, 2000, 3, 1

dtype = np.float32

np.random.seed(0)
x = np.random.rand(M, 1, D).astype(dtype) / math.sqrt(D)
y = np.random.rand(1, N, D).astype(dtype) / math.sqrt(D)
b = np.random.randn(N, DV).astype(dtype)
a = np.empty((M, DV), dtype=dtype)


def fun(x, y, b, backend, out=None):
    if "keops" in backend:
        x = LazyTensor(x)
        y = LazyTensor(y)
    Dxy = ((x - y)).sum(axis=2)
    if backend == "keops":
        Kxy = (-Dxy).exp()
        out = Kxy.__matmul__(b, out=out)
    else:
        Kxy = np.exp(-Dxy)
        out = Kxy @ b
    return out


backends = ["keops", "numpy"]

out = []
for backend in backends:
    out.append(fun(x, y, b, backend, out=a).squeeze())


def test_lazytensor_gaussian_numpy_inplace():
    assert_np_allclose(out[0], out[1], label="gaussian_numpy_inplace")
