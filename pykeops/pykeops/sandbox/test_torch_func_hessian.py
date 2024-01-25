import torch
from pykeops.torch import LazyTensor

import keopscore

keopscore.auto_factorize = False


def fn_torch(x_i):
    K_ij = (x_i**2).sum(-1).exp()
    K_ij = K_ij[..., None]
    res = K_ij.sum(2)
    res = res.sum()
    return res


def fn_keops(x_i):
    x_i = LazyTensor(x_i)
    K_ij = (x_i**2).sum(-1).exp()
    res = K_ij.sum(2)
    res = res.sum()
    return res


ntest = 10
err = []
B1, B2, M, N, D = 5, 4, 30, 3, 2

hessian = torch.func.hessian
# hessian = lambda f : torch.func.jacrev(torch.func.jacrev(f))

for k in range(ntest):
    x_i = torch.ones(B1, B2, M, 1, D)
    y_j = torch.ones(B1, B2, 1, N, D)
    b_j = torch.ones(B1, B2, 1, N, 1)
    p = torch.ones(B1, B2, 1, 1, 1)
    res1 = hessian(fn_torch)(x_i)
    res2 = hessian(fn_keops)(x_i)
    err.append((torch.norm(res1 - res2) / torch.norm(res1)).item())

print("mean=", sum(err) / ntest)
print("max=", max(err))
print("err=", torch.log10(torch.tensor(err)))


class TestCase:
    def test_torch_func_hessian(self):
        res1 = torch.func.hessian(fn_torch)(x_i)
        res2 = torch.func.hessian(fn_keops)(x_i)
        assert torch.allclose(res1, res2)
