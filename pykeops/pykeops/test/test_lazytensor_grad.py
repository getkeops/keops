import torch
import pykeops.config

from pykeops.torch import LazyTensor
from pykeops.test import assert_torch_allclose

M, N, D, DV = 100, 100, 3, 1

dtype = torch.float32

torch.backends.cuda.matmul.allow_tf32 = False
use_cuda = torch.cuda.is_available() and pykeops.config.cuda.is_available()
device_id = "cuda" if use_cuda else "cpu"

torch.manual_seed(0)
x = torch.rand(M, 1, D, requires_grad=True, device=device_id, dtype=dtype)
y = torch.rand(1, N, 1, device=device_id, dtype=dtype)
b = torch.randn(N, DV, device=device_id, dtype=dtype)


def fun(x, y, b, backend):
    if "keops" in backend:
        x = LazyTensor(x)
        y = LazyTensor(y)
    # Kxy = ((x - 0.5).mod(1, 0.2) - y).sum(dim=2)
    Kxy = (x.cos() - y).sum(dim=2)
    out = Kxy @ b
    if device_id != "cpu":
        torch.cuda.synchronize()
    # print("out:",out.flatten()[:10])
    return out


backends = ["keops", "torch"]  # "keops_old"

out = []
for backend in backends:
    out.append(fun(x, y, b, backend).squeeze())

out_g = []
for k, backend in enumerate(backends):
    out_g.append(torch.autograd.grad((out[k] ** 2).sum(), [x], create_graph=True)[0])

out_g2 = []
for k, backend in enumerate(backends):
    out_g2.append(torch.autograd.grad((out_g[k] ** 2).sum(), [x])[0])


class TestClass:
    def test_lazytensor_grad(self):
        assert_torch_allclose(out[0], out[1], rtol=0.0001, label="lazytensor_grad_fw")

    def test_lazytensor_grad_bw1(self):
        assert_torch_allclose(
            out_g[0], out_g[1], rtol=0.0001, label="lazytensor_grad_bw1"
        )

    def test_lazytensor_grad_bw2(self):
        assert_torch_allclose(
            out_g2[0], out_g2[1], rtol=0.001, label="lazytensor_grad_bw2"
        )
