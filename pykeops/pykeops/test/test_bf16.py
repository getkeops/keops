import pytest
import torch
from pykeops.torch import LazyTensor

dtype = torch.bfloat16

M, N, D = 5, 5, 1

torch.backends.cuda.matmul.allow_tf32 = False
device_id = "cuda" if torch.cuda.is_available() else "cpu"


def create_test_tensors():
    torch.manual_seed(0)
    x = torch.randn(M, 1, D, dtype=dtype, requires_grad=True, device="cuda")
    y = torch.randn(1, N, D, dtype=dtype, device="cuda")
    return x, y


def fun(x, y, backend):
    if backend == "keops":
        x = LazyTensor(x)
        y = LazyTensor(y)
    elif backend != "torch":
        raise ValueError("wrong backend")
    Dxy = (x - y).sum(dim=2)
    Kxy = Dxy
    return Kxy.sum(dim=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
def test_bfloat16_fw():
    x, y = create_test_tensors()
    out = []
    for backend in ["torch", "keops"]:
        out.append(fun(x, y, backend).squeeze())

    assert torch.allclose(out[0], out[1], atol=0.01, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
def test_bfloat16_bw():
    x, y = create_test_tensors()
    out = []
    for backend in ["torch", "keops"]:
        out.append(fun(x, y, backend).squeeze())

    out_g = []
    for k, backend in enumerate(["torch", "keops"]):
        out_g.append(torch.autograd.grad(out[k][0], [x])[0])

    assert torch.allclose(out_g[0], out_g[1], atol=0.01, rtol=0.01)
