import pytest
import torch
from pykeops.torch import LazyTensor


def gaussian_sum(x, y):
    xi = LazyTensor(x[:, None, :])  # Vi(D)
    yj = LazyTensor(y[None, :, :])  # Vj(D)
    K = ((xi - yj) ** 2).sum(-1)
    return K


@pytest.mark.parametrize("M,N,D", [(10, 15, 3), (32, 40, 5)])
def test_slice_value_parity(M, N, D):
    torch.manual_seed(0)
    x = torch.randn(M, D, requires_grad=True)
    y = torch.randn(N, D, requires_grad=True)

    # Reference computation using pure PyTorch (no KeOps involved):
    # squared Euclidean distance matrix between all x_i and y_j
    K_full = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)

    sli = slice(2, 8)
    slj = slice(3, 14, 1)
    K_slice_ref = K_full[sli, slj]

    K_symbolic = gaussian_sum(x, y)
    K_slice_sym = K_symbolic[sli, slj]

    # Evaluate both reference and symbolic slices through explicit reductions
    out_ref = K_slice_ref.sum(axis=1).sum(axis=0)

    out_sym = K_slice_sym.sum(axis=1).sum(axis=0)

    # Ensure we work with plain tensors
    if hasattr(out_ref, "__GenericLazyTensor__"):
        out_ref = out_ref()
    if hasattr(out_sym, "__GenericLazyTensor__"):
        out_sym = out_sym()

    assert torch.allclose(out_sym, out_ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("M,N,D", [(6, 7, 3)])
def test_slice_gradcheck(M, N, D):
    torch.manual_seed(1)
    x = torch.randn(M, D, dtype=torch.double, requires_grad=True)
    y = torch.randn(N, D, dtype=torch.double, requires_grad=True)

    sli = slice(1, 5)
    slj = slice(2, 6)

    def func(x_, y_):
        K = gaussian_sum(x_, y_)[sli, slj]
        # Use a numerically-stable kernel to avoid huge gradients, very important !!
        return (-K).exp().sum(axis=1).sum(axis=0)

    assert torch.autograd.gradcheck(func, (x, y), eps=1e-6, atol=1e-4, rtol=1e-3)
