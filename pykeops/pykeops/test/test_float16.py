# Test for Clamp operation using LazyTensors
import pytest
import torch
from pykeops.torch import LazyTensor

dtype = torch.float16

M, N, D = 5, 5, 1

torch.backends.cuda.matmul.allow_tf32 = False
device_id = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
x = torch.randn(M, 1, D, dtype=dtype, requires_grad=True, device=device_id)
y = torch.randn(1, N, D, dtype=dtype, device=device_id)


def fun(x, y, backend):
    if backend == "keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
    elif backend != "torch":
        raise ValueError("wrong backend")
    Dxy = (x - y).sum(dim=2)
    Kxy = Dxy
    return Kxy.sum(dim=0)


class TestCase:
    out = []

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
    def test_float16_fw(self):
        for backend in ["torch", "keops"]:
            self.out.append(fun(x, y, backend).squeeze())

        assert torch.allclose(self.out[0], self.out[1], atol=0.001, rtol=0.001)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
    def test_float16_bw(self):
        out_g = []
        for k, backend in enumerate(["torch", "keops"]):
            out_g.append(torch.autograd.grad(self.out[k][0], [x])[0])

        assert torch.allclose(out_g[0], out_g[1])
