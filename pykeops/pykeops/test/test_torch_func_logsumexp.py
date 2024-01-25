import torch
from pykeops.torch import LazyTensor

torch.manual_seed(0)

x_i = torch.randn(5, 4, 10, 1, 2)
y_j = torch.randn(5, 1, 1, 20, 2)
b_j = torch.rand(5, 4, 1, 20, 1)
dx_i = torch.randn(5, 4, 10, 1, 2)
dy_j = torch.randn(5, 1, 1, 20, 2)
db_j = torch.randn(5, 4, 1, 20, 1)


def fn_torch(x_i, y_j, b_j):
    D2_ij = ((x_i - y_j) ** 2).sum(-1)
    D2_ij = D2_ij[..., None]
    return ((D2_ij.exp() * b_j).sum(2)).log().norm()


def fn_keops(x_i, y_j, b_j):
    x_i = LazyTensor(x_i)
    y_j = LazyTensor(y_j)
    b_j = LazyTensor(b_j)
    D2_ij = ((x_i - y_j) ** 2).sum(-1)
    return D2_ij.logsumexp(axis=2, other=b_j).norm()


class TestCase:
    def test_torch_func_vmap(self):
        res1 = torch.func.vmap(fn_torch)(x_i, y_j, b_j)
        res2 = torch.func.vmap(fn_keops)(x_i, y_j, b_j)
        assert torch.allclose(res1, res2, atol=1e-5)

    def test_torch_func_grad(self):
        res1 = torch.func.grad(fn_torch, (0, 1, 2))(x_i, y_j, b_j)
        res2 = torch.func.grad(fn_keops, (0, 1, 2))(x_i, y_j, b_j)
        for k in range(3):
            assert torch.allclose(res1[k], res2[k], atol=1e-5)

    def test_torch_func_vjp(self):
        res1 = torch.func.vjp(fn_torch, x_i, y_j, b_j)[1](torch.tensor(1.0))[0]
        res2 = torch.func.vjp(fn_keops, x_i, y_j, b_j)[1](torch.tensor(1.0))[0]
        assert torch.allclose(res1, res2, atol=1e-5)

    def test_torch_func_jvp(self):
        _, res1 = torch.func.jvp(fn_torch, (x_i, y_j, b_j), (dx_i, dy_j, db_j))
        _, res2 = torch.func.jvp(fn_keops, (x_i, y_j, b_j), (dx_i, dy_j, db_j))
        assert torch.allclose(res1, res2, atol=1e-5)

    def test_torch_func_jacrev(self):
        res1 = torch.func.jacrev(fn_torch, (0, 1, 2))(x_i, y_j, b_j)
        res2 = torch.func.jacrev(fn_keops, (0, 1, 2))(x_i, y_j, b_j)
        for k in range(3):
            assert torch.allclose(res1[k], res2[k], atol=1e-5)

    def test_torch_func_jacfwd(self):
        res1 = torch.func.jacfwd(fn_torch)(x_i, y_j, b_j)
        res2 = torch.func.jacfwd(fn_keops)(x_i, y_j, b_j)
        assert torch.allclose(res1, res2, atol=1e-5)

    def test_torch_func_hessian(self):
        res1 = torch.func.hessian(fn_torch)(x_i, y_j, b_j)
        res2 = torch.func.hessian(fn_keops)(x_i, y_j, b_j)
        assert torch.allclose(res1, res2, atol=1e-5)
