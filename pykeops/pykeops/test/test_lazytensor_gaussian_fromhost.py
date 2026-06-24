import math
import torch
import pykeops.config
from pykeops.torch import LazyTensor
from pykeops.test import assert_torch_allclose

M, N, D, DV = 2500, 2000, 3, 1

dtype = torch.float64
sum_scheme = "block_sum"

torch.backends.cuda.matmul.allow_tf32 = False
use_cuda = torch.cuda.is_available() and pykeops.config.cuda.is_available()
device_id = 0 if use_cuda else -1

torch.manual_seed(0)
x = torch.rand(M, 1, D, dtype=dtype) / math.sqrt(D)
y = torch.rand(1, N, D, dtype=dtype) / math.sqrt(D)
b = torch.randn(N, DV, dtype=dtype)


def fun(x, y, b, backend):
    if "keops" in backend:
        x = LazyTensor(x)
        y = LazyTensor(y)
    Dxy = ((x - y).square()).sum(dim=2)
    Kxy = (-Dxy).exp()
    if "keops" in backend:
        out = Kxy.__matmul__(b, sum_scheme=sum_scheme, device_id=device_id)
    else:
        out = Kxy @ b
    # print("out:",out)
    return out


backends = ["keops", "torch"]

out = []
for backend in backends:
    out.append(fun(x, y, b, backend).squeeze())


def test_lazytensor_gaussian_fromhost():
    assert_torch_allclose(out[0], out[1], label="gaussian_fromhost")
