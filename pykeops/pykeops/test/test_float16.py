# Test for Clamp operation using LazyTensors
import pytest
import torch
import pykeops.config
from pykeops.torch import LazyTensor
from pykeops.test import assert_torch_allclose

dtype = torch.float16

M, N, D = 5, 5, 1

torch.backends.cuda.matmul.allow_tf32 = False
use_cuda = torch.cuda.is_available() and pykeops.config.cuda.is_available()
device_id = "cuda" if use_cuda else "cpu"

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

    @pytest.mark.skipif(
        not use_cuda,
        reason="Requires torch and KeOps CUDA support",
    )
    def test_float16_fw(self):
        for backend in ["torch", "keops"]:
            self.out.append(fun(x, y, backend).squeeze())

        assert_torch_allclose(
            self.out[0], self.out[1], atol=0.001, rtol=0.001, label="float16_fw"
        )

    @pytest.mark.skipif(
        not use_cuda,
        reason="Requires torch and KeOps CUDA support",
    )
    def test_float16_bw(self):
        out_g = []
        for k, backend in enumerate(["torch", "keops"]):
            out_g.append(torch.autograd.grad(self.out[k][0], [x])[0])

        assert_torch_allclose(out_g[0], out_g[1], label="float16_bw")


@pytest.mark.skipif(not use_cuda, reason="Requires torch and KeOps CUDA support")
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("m,n", [(5, 6), (6, 5)])
def test_sparse_half2(axis, m, n):
    from pykeops.torch import Genred
    from pykeops.torch.cluster import from_matrix

    ri = torch.tensor([[0, 1], [1, m]], device="cuda")
    rj = torch.tensor([[0, 2], [2, 3], [3, n]], device="cuda")
    keep = torch.tensor([[True, False, True], [False, True, False]], device="cuda")
    ranges = from_matrix(ri, rj, keep)
    original = [a.clone() for a in ranges]
    mask = keep.repeat_interleave(ri[:, 1] - ri[:, 0], dim=0).repeat_interleave(
        rj[:, 1] - rj[:, 0], dim=1
    )

    def inputs(rows):
        values = torch.arange(2 * rows, device="cuda", dtype=torch.float16).reshape(
            rows, 2
        )
        return ((values % 7 - 3) / 2).requires_grad_()

    x, y = inputs(m), inputs(n)
    p = torch.tensor([0.5, 1.0], dtype=torch.float16, device="cuda", requires_grad=True)
    ref = ((x[:, None] - y[None, :]).square() * p * mask[..., None]).sum(axis)
    op = Genred(
        "p * Square(x-y)",
        ["x = Vi(2)", "y = Vj(2)", "p = Pm(2)"],
        reduction_op="Sum",
        axis=axis,
    )
    result = op(x, y, p, ranges=ranges)
    torch.testing.assert_close(result, ref, atol=0, rtol=0)
    adjoint = (
        torch.arange(result.numel(), device="cuda").reshape_as(result) % 3 - 1
    ).to(result)
    actual = torch.autograd.grad(result, (x, y, p), adjoint)
    expected = torch.autograd.grad(ref, (x, y, p), adjoint)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(ranges, original):
        torch.testing.assert_close(a, b)
    buf = torch.empty_like(ref)
    with torch.no_grad():
        returned = op(x, y, p, ranges=ranges, out=buf)
    assert returned.data_ptr() == buf.data_ptr()
    torch.testing.assert_close(buf, ref, atol=0, rtol=0)
