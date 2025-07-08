from __future__ import annotations

import itertools
from typing import Callable, Tuple

import pytest
import torch
from pykeops.torch import LazyTensor

# -----------------------------------------------------------------------------
# Global config
# -----------------------------------------------------------------------------
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Disable TF32 by default (will be toggled in a dedicated test below)
torch.backends.cuda.matmul.allow_tf32 = False

# Default tolerances for bf16 comparisons
default_rtol = 3e-2  # 3 % relative error
default_atol = 3e-2  # 3 % absolute error

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def rand_tensor(shape: Tuple[int, ...], *, scale: float = 1.0) -> torch.Tensor:
    """Random bf16 tensor of the given *shape* and *scale*."""
    return (torch.randn(shape, dtype=torch.float32, device=device) * scale).to(dtype)


def to_bf16(t: torch.Tensor) -> torch.Tensor:
    """Cast tensor to bf16."""
    return t.to(dtype)


def assert_close(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Wrapper around *torch.allclose* with nicer error messages."""
    rtol = default_rtol if rtol is None else rtol
    atol = default_atol if atol is None else atol

    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        delta = (a - b).abs()
        raise AssertionError(
            f"max abs err {delta.max():.3e}, max rel err {(delta / a.abs().clamp_min(1)).max():.3e}"
        )


# -----------------------------------------------------------------------------
# Kernels test
# -----------------------------------------------------------------------------


def k_sum(x: torch.Tensor, y: torch.Tensor, *, backend: str) -> torch.Tensor:
    """Sum kernel → output shape (N,).

    Matches the definition used in *test_bf16.py*: we sum over the feature
    dimension (D) then over the *i*-index (M).  We purposely keep the *j*
    dimension (N) so the output has size ``(N,)`` – this avoids inadvertent
    double-counting that happened when we summed over both *i* and *j*.
    """
    if backend == "keops":
        x, y = LazyTensor(x), LazyTensor(y)

    return (x - y).sum(dim=-1).sum(dim=0).squeeze()


def k_exp_sqnorm(x: torch.Tensor, y: torch.Tensor, *, backend: str) -> torch.Tensor:
    """RBF-like kernel → output shape (N,)."""
    if backend == "keops":
        x, y = LazyTensor(x), LazyTensor(y)

    d2 = ((x - y) ** 2).sum(dim=-1)  # shape (M, N)
    return (-d2).exp().sum(dim=0).squeeze()


ALL_FUNS: list[Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor]] = [
    k_sum,
    k_exp_sqnorm,
]

# -----------------------------------------------------------------------------
# Reference vs Keops helper
# -----------------------------------------------------------------------------


def reference_and_keops(
    fun: Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
):
    ref = fun(to_bf16(x), to_bf16(y), backend="torch")
    ko = fun(to_bf16(x), to_bf16(y), backend="keops")
    return ref, ko


# -----------------------------------------------------------------------------
# Parameter grids – moderate sizes
# -----------------------------------------------------------------------------
M_VALUES = [1, 5, 25]
N_VALUES = [1, 5, 25]
D_VALUES = [1, 4, 16]
SHAPES = list(itertools.product(M_VALUES, N_VALUES, D_VALUES))
SHAPE_IDS = [f"M{m}_N{n}_D{d}" for (m, n, d) in SHAPES]

# Scaling factors (keep within representable range)
SCALES = {"unit": 1.0, "tiny": 1e-2, "huge": 1e2}


# -----------------------------------------------------------------------------
# Forward tests
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
@pytest.mark.parametrize("fun", ALL_FUNS)
@pytest.mark.parametrize("M,N,D", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("scale_key", list(SCALES))
def test_forward(fun, M: int, N: int, D: int, scale_key: str):
    scale = SCALES[scale_key]
    x = rand_tensor((M, 1, D), scale=scale).requires_grad_(True)
    y = rand_tensor((1, N, D), scale=scale)
    # ------------------------------------------------------------------
    # Adaptive tolerances: bf16 has ε ≈ 2**-7 ≃ 7.8e-3.
    # ------------------------------------------------------------------
    bf16_eps = 2**-7  # ≈7.8e-3
    terms_sqrt = (M * D) ** 0.5

    rtol = max(default_rtol, 2.0 * bf16_eps * terms_sqrt)
    atol = max(default_atol, 10.0 * bf16_eps * scale * terms_sqrt)

    ref, ko = reference_and_keops(fun, x, y)
    assert_close(ref, ko, rtol=rtol, atol=atol)


# -----------------------------------------------------------------------------
# Backward tests on a subset of shapes
# -----------------------------------------------------------------------------
BACKWARD_SHAPES = [(5, 5, 4), (25, 7, 1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
@pytest.mark.parametrize("fun", ALL_FUNS)
@pytest.mark.parametrize("M,N,D", BACKWARD_SHAPES)
def test_backward(fun, M: int, N: int, D: int):
    x = rand_tensor((M, 1, D)).requires_grad_(True)
    y = rand_tensor((1, N, D))
    ref, ko = reference_and_keops(fun, x, y)
    grad = torch.randn_like(ref)
    (g_ref,) = torch.autograd.grad(ref, x, grad_outputs=grad)
    (g_ko,) = torch.autograd.grad(ko, x, grad_outputs=grad)
    assert_close(g_ref, g_ko, rtol=5e-2, atol=5e-2)


# -----------------------------------------------------------------------------
# Gradcheck – double precision, tiny shape
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
@pytest.mark.parametrize("fun", ALL_FUNS)
def test_gradcheck(fun):
    x = torch.randn(3, 1, 2, dtype=torch.float64, device=device, requires_grad=True)
    y = torch.randn(1, 4, 2, dtype=torch.float64, device=device)
    torch.autograd.gradcheck(
        lambda u: fun(u, y, backend="keops"), (x,), eps=1e-6, atol=1e-3, rtol=1e-3
    )


# -----------------------------------------------------------------------------
# Small-shape tests
# -----------------------------------------------------------------------------

SMALL_SHAPES = [(1, 1, 1), (2, 3, 2), (3, 4, 3), (10, 10, 4)]
SMALL_IDS = [f"M{m}_N{n}_D{d}" for (m, n, d) in SMALL_SHAPES]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
@pytest.mark.parametrize("M,N,D", SMALL_SHAPES, ids=SMALL_IDS)
def test_small_sum(M: int, N: int, D: int):
    """Quick sanity check on small random shapes."""
    x = rand_tensor((M, 1, D)).requires_grad_(True)
    y = rand_tensor((1, N, D))
    ref, ko = reference_and_keops(k_sum, x, y)
    assert_close(ref, ko)


# -----------------------------------------------------------------------------
# TF32 interaction test
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
@pytest.mark.parametrize("tf32", [False, True])
def test_tf32_flag(tf32: bool):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    x = rand_tensor((10, 1, 4)).requires_grad_(True)
    y = rand_tensor((1, 12, 4))
    ref, ko = reference_and_keops(k_sum, x, y)
    assert_close(ref, ko)
    torch.backends.cuda.matmul.allow_tf32 = False  # reset
