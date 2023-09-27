import numpy as np
import torch

from pykeops.numpy import Genred

M = 11
N = 150
axis = 1

############################################################################

dimfa = [3, 2, 1, 2]
dimfb = [1, 2, 2, 3]

x = np.random.rand(M, np.array(dimfa).prod())
y = np.random.rand(N, np.array(dimfb).prod())

gamma_py = np.zeros((M, N, np.array(dimfa).prod() * np.array(dimfb).prod()))
for i in range(M):
    for j in range(N):
        gamma_py[i, j, :] = np.kron(
            x[i, :].reshape(*dimfa), y[j, :].reshape(*dimfb)
        ).flatten()

gamma_py = gamma_py.sum(axis=axis)


def test_kron_lazytensor_np():
    from pykeops.numpy import LazyTensor

    X = LazyTensor(x[:, None, :])
    Y = LazyTensor(y[None, :, :])
    gamma_keops_lazytensor_np = (X.keops_kron(Y, dimfa, dimfb)).sum(axis=axis)

    assert np.allclose(gamma_keops_lazytensor_np, gamma_py, atol=1e-6)


############################################################################
def test_kron_lazytensor_torch():
    from pykeops.torch import LazyTensor as LazyTensor_torch

    X_t = LazyTensor_torch(torch.from_numpy(x)[:, None, :])
    Y_t = LazyTensor_torch(torch.from_numpy(y)[None, :, :])
    gamma_keops_lazytensor_torch = (X_t.keops_kron(Y_t, dimfa, dimfb)).sum(axis=axis)

    assert torch.allclose(
        gamma_keops_lazytensor_torch, torch.from_numpy(gamma_py), atol=1e-6
    )


############################################################################
Ddx = 2
Ddx2 = 3
Dx = Ddx * Ddx2
Dy = Ddx * Ddx

x0 = np.random.rand(M, Dx)
y0 = np.random.rand(N, Dy)

gamma_py2 = np.zeros((M, N, Dx * Dy))
for i in range(M):
    for j in range(N):
        gamma_py2[i, j, :] = np.kron(
            x0[i, :].reshape(Ddx, Ddx2), y0[j, :].reshape(Ddx, Ddx)
        ).flatten()

gamma_py2 = gamma_py2.sum(axis=axis)


def test_kron_genred():
    aliases = [f"x=Vi(0,{Dx})", f"y=Vj(1,{Dy})"]
    formula = f"Kron(x, y, [{Ddx}, {Ddx2}], [{Ddx}, {Ddx}])"
    formula2 = f"TensorDot(x, y, [{Ddx}, {Ddx2}], [{Ddx}, {Ddx}], [], [], [0, 2, 1, 3])"

    # Call cuda kernel
    myconv = Genred(formula, aliases, reduction_op="Sum", axis=axis)
    gamma_keops_genred_Kron = myconv(x0, y0)

    myconv2 = Genred(formula2, aliases, reduction_op="Sum", axis=axis)
    gamma_keops_genred_TensorDot = myconv2(x0, y0)

    assert np.allclose(gamma_keops_genred_Kron, gamma_py2, atol=1e-6)
    assert np.allclose(gamma_keops_genred_TensorDot, gamma_py2, atol=1e-6)


############################################################################

Dx2 = Ddx * Ddx2 * Ddx
Dy2 = Ddx * Ddx * Ddx2

x2 = np.random.rand(M, Dx2)
y2 = np.random.rand(N, Dy2)

gamma_py3 = np.zeros((M, N, Dx2 * Dy2))
for i in range(M):
    for j in range(N):
        gamma_py3[i, j, :] = np.kron(
            x2[i, :].reshape(Ddx, Ddx2, Ddx), y2[j, :].reshape(Ddx, Ddx, Ddx2)
        ).flatten()

gamma_py3 = gamma_py3.sum(axis=axis)

gamma_py4 = np.zeros((M, N, Dx2 * Dy2))
for i in range(M):
    for j in range(N):
        gamma_py4[i, j, :] = np.einsum(
            "ijk, lmn -> iljmkn",
            x2[i, :].reshape(Ddx, Ddx2, Ddx),
            y2[j, :].reshape(Ddx, Ddx, Ddx2),
        ).flatten()

gamma_py4 = gamma_py4.sum(axis=axis)


def test_kron_genred2():
    aliases = [f"x=Vi(0,{Dx2})", f"y=Vj(1,{Dy2})"]
    formula = f"Kron(x, y, [{Ddx}, {Ddx2}, {Ddx}], [{Ddx}, {Ddx}, {Ddx2}])"
    formula2 = f"TensorDot(x, y, [{Ddx}, {Ddx2}, {Ddx}], [{Ddx}, {Ddx}, {Ddx2}], [], [], [0, 2, 4, 1, 3, 5])"

    myconv = Genred(formula, aliases, reduction_op="Sum", axis=axis)
    gamma_keops_genred_Kron2 = myconv(x2, y2)

    myconv2 = Genred(formula2, aliases, reduction_op="Sum", axis=axis)
    gamma_keops_genred_TensorDot2 = myconv2(x2, y2)

    assert np.allclose(gamma_keops_genred_Kron2, gamma_py3, atol=1e-6)
    assert np.allclose(gamma_keops_genred_TensorDot2, gamma_py3, atol=1e-6)
    assert np.allclose(gamma_py4, gamma_py3, atol=1e-6)
