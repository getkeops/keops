import math
import unittest
import torch
from pykeops.torch import LazyTensor
from pykeops.test import assert_torch_allclose

M, N, D, DV = 2000, 3000, 3, 1

dtype = torch.float32

torch.manual_seed(42)

torch.backends.cuda.matmul.allow_tf32 = False
device_id = "cuda" if torch.cuda.is_available() else "cpu"


def fun(x, y, b, backend):
    if "keops" in backend:
        x = LazyTensor(x)
        y = LazyTensor(y)
    Dxy = ((x - y) ** 2).sum(dim=2)
    Kxy = (-Dxy).exp()
    if backend == "keops2D":
        out = LazyTensor.__matmul__(Kxy, b, backend="GPU_2D")
    else:
        out = Kxy @ b
    if device_id != "cpu":
        torch.cuda.synchronize()
    # print("out:",out)
    return out


backends = ["keops2D", "torch"]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = torch.rand(M, 1, D, device=device_id, dtype=dtype) / math.sqrt(D)
        y = torch.rand(1, N, D, device=device_id, dtype=dtype) / math.sqrt(D)
        b = torch.randn(N, DV, requires_grad=True, device=device_id, dtype=dtype)
        cls.out = []
        for backend in backends:
            cls.out.append(fun(x, y, b, backend).squeeze())

        cls.out_g = []
        for k in range(len(backends)):
            cls.out_g.append(
                torch.autograd.grad((cls.out[k] ** 2).sum(), [b], create_graph=True)[0]
            )

        cls.out_g2 = []
        for k in range(len(backends)):
            cls.out_g2.append(torch.autograd.grad((cls.out_g[k] ** 2).sum(), [b])[0])

    def test_conv2d_fw(self):
        assert_torch_allclose(self.out[0], self.out[1], label="conv2d_fw")

    def test_conv2d_bw1(self):
        assert_torch_allclose(self.out_g[0], self.out_g[1], label="conv2d_bw1")

    def test_conv2d_bw2(self):
        assert_torch_allclose(self.out_g2[0], self.out_g2[1], label="conv2d_bw2")
