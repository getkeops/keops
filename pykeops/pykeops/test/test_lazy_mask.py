import torch
import pytest
from pykeops.torch import LazyTensor


def gaussian_sum(x, y):
    xi = LazyTensor(x[:, None, :])
    yj = LazyTensor(y[None, :, :])
    K = ((xi - yj) ** 2).sum(-1)
    return K


@pytest.mark.parametrize("M,N,D", [(10, 15, 3), (32, 40, 5)])
def test_mask_value_parity(M, N, D):
    torch.manual_seed(0)
    x = torch.randn(M, D, requires_grad=True)
    y = torch.randn(N, D, requires_grad=True)

    # random list of K index pairs
    K = min(M * N // 3, 30)
    I = torch.randint(0, M, (K,), dtype=torch.long)
    J = torch.randint(0, N, (K,), dtype=torch.long)

    K_full = gaussian_sum(x, y)()
    ref = K_full[I, J].sum()  # eager gather then reduction

    K_sym = gaussian_sum(x, y)
    sym = K_sym.masked_select(I, J).sum(axis=1).sum(axis=0)

    assert torch.allclose(sym(), ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("M,N,D", [(6, 7, 3)])
def test_mask_gradcheck(M, N, D):
    torch.manual_seed(1)
    x = torch.randn(M, D, dtype=torch.double, requires_grad=True)
    y = torch.randn(N, D, dtype=torch.double, requires_grad=True)

    K = 12
    I = torch.randint(0, M, (K,), dtype=torch.long)
    J = torch.randint(0, N, (K,), dtype=torch.long)

    def func(x_, y_):
        Kmat = gaussian_sum(x_, y_)
        out = (-Kmat).masked_select(I, J).exp().sum(axis=1).sum(axis=0)
        return out()

    assert torch.autograd.gradcheck(func, (x, y), eps=1e-6, atol=1e-4, rtol=1e-3) 