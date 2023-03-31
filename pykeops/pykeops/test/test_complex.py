# Non-Uniform Discrete Fourier Tranform example

import math
import torch
from pykeops.torch import LazyTensor

dtype = torch.float32
dtype_c = torch.complex64

M, N, D = 1000, 1000, 1

torch.backends.cuda.matmul.allow_tf32 = False
device_id = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
x = torch.rand(1, N, D, dtype=dtype_c, requires_grad=True, device=device_id)
p = torch.rand(1, N, D, dtype=dtype, device=device_id)
f = torch.rand(M, 1, D, dtype=dtype, device=device_id)


def view_as_real(x):
    if torch.is_complex(x):
        return torch.view_as_real(x)
    else:
        return x


def fun(x, p, f, backend):
    if "keops" in backend:
        x = LazyTensor(x)
        p = LazyTensor(p)
        f = LazyTensor(f)
    X = x * (-2 * math.pi * 1j * p * f).exp()
    return X.sum(dim=0)


out = []
for backend in ["keops", "torch"]:
    out.append(fun(x, p, f, backend).squeeze())


def test_complex_fw():
    assert torch.allclose(out[0], out[1])


# out_g = []
# for k, backend in enumerate(["keops", "torch"]):
#     if out[k].is_complex():
#         out_g.append(
#             torch.autograd.grad((out[k].real ** 2 + out[k].imag ** 2).sum(), [x])[0]
#         )
#     else:
#         out_g.append(torch.autograd.grad((out[k] ** 2).sum(), [x])[0])
#
#
# def test_complex_bw():
#     assert torch.allclose(out_g[0], out_g[1])
