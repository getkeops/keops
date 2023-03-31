import numpy as np
from pykeops.numpy import LazyTensor, ComplexLazyTensor

M, N, D = 1000, 1000, 3

dtype = "float32"

np.random.seed(0)
x = np.random.rand(M, 1, D).astype(dtype) + 1j * np.random.rand(M, 1, D).astype(dtype)
y = np.random.rand(1, N, D).astype(dtype) + 1j * np.random.rand(1, N, D).astype(dtype)


def fun(x, y, backend):
    if backend == "keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
    Kxy = ((x * y) * y.real + x + x.real).sum(axis=2)
    return Kxy.sum(axis=0)


out = []
for backend in ["numpy", "keops"]:
    out.append(fun(x, y, backend).squeeze())


def test_complex_numpy():
    assert np.allclose(out[0], out[1])
