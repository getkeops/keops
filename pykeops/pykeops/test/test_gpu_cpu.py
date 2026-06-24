import math
import pytest
import torch
import pykeops.config
from pykeops.torch import LazyTensor

use_cuda = pykeops.config.cuda.is_available()

M, N, D, DV = 1000, 1000, 3, 1

dtype = torch.float32
device_id = "cpu"

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)

x = torch.rand(M, 1, D, device=device_id, dtype=dtype) / math.sqrt(D)
y = torch.rand(1, N, D, device=device_id, dtype=dtype) / math.sqrt(D)
b = torch.randn(N, DV, device=device_id, dtype=dtype)


def fun(x, y, b, backend):
    if "keops" in backend:
        x = LazyTensor(x)
        y = LazyTensor(y)
    Dxy = ((x - y) ** 2).sum(dim=2)
    Kxy = (-Dxy).exp()
    if "keops" in backend:
        if backend.split("_")[1] == "gpu":
            out = Kxy.__matmul__(b, backend="GPU_1D")
        elif backend.split("_")[1] == "cpu":
            out = Kxy.__matmul__(b, backend="CPU")
    else:
        out = Kxy @ b
    return out


out = []
for backend in ["torch", "keops_cpu"]:
    out.append(fun(x, y, b, backend).squeeze())


class TestCase:
    def test_torch_keops_cpu(self):
        assert torch.allclose(
            out[0], out[1]
        ), f"torch vs keops_cpu mismatch: ||ref-test||_2={torch.norm(out[0] - out[1]).item():.6e} and ||ref||_2={torch.norm(out[0]).item():.6e}"

    @pytest.mark.skipif(
        not use_cuda,
        reason="Requires KeOps CUDA support",
    )
    def test_torch_keops_gpu(self):
        out_gpu = fun(x, y, b, ["keops_gpu"]).squeeze()
        assert torch.allclose(
            out[0], out_gpu
        ), f"torch vs keops_gpu mismatch: ||ref-test||_2={torch.norm(out[0] - out_gpu).item():.6e} and ||ref||_2={torch.norm(out[0]).item():.6e}"
