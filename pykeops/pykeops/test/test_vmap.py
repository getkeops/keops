import torch
from pykeops.torch import LazyTensor

test_grad = True
torch.manual_seed(0)


def fn(x_i, x_j, y_j, use_keops=True):
    if use_keops:
        x_i = LazyTensor(x_i)
        x_j = LazyTensor(x_j)
        y_j = LazyTensor(y_j)
    K_ij = (-((x_i - x_j) ** 2).sum(-1)).exp()
    if not use_keops:
        K_ij = K_ij[..., None]
    return (K_ij * y_j).sum(1)


class TestCase:
    def test_vmap_0(self):
        x_i = torch.randn(5, 10, 1, 2, requires_grad=test_grad, dtype=torch.float64)
        x_j = torch.randn(5, 1, 20, 2, dtype=torch.float64)
        y_j = torch.randn(5, 1, 20, 1, dtype=torch.float64)
        in_dims = 0
        out_dims = 0

        res_keops = torch.vmap(fn, in_dims=in_dims, out_dims=out_dims)(
            x_i, x_j, y_j, use_keops=True
        )
        res_torch = torch.vmap(fn, in_dims=in_dims, out_dims=out_dims)(
            x_i, x_j, y_j, use_keops=False
        )

        assert torch.allclose(res_keops, res_torch)

        u = torch.rand(res_torch.shape)
        (res_torch_grad,) = torch.autograd.grad(res_torch, x_i, u)
        (res_keops_grad,) = torch.autograd.grad(res_keops, x_i, u)

        assert torch.allclose(res_keops_grad, res_torch_grad)

    def test_vmap_1(self):
        x_i = torch.randn(10, 5, 1, 2, requires_grad=test_grad, dtype=torch.float64)
        x_j = torch.randn(1, 5, 20, 2, dtype=torch.float64)
        y_j = torch.randn(1, 5, 20, 1, dtype=torch.float64)
        in_dims = 1
        out_dims = 1

        res_keops = torch.vmap(fn, in_dims=in_dims, out_dims=out_dims)(
            x_i, x_j, y_j, use_keops=True
        )
        res_torch = torch.vmap(fn, in_dims=in_dims, out_dims=out_dims)(
            x_i, x_j, y_j, use_keops=False
        )
        assert torch.allclose(res_keops, res_torch)

        u = torch.rand(res_torch.shape)
        (res_torch_grad,) = torch.autograd.grad(res_torch, x_i, u)
        (res_keops_grad,) = torch.autograd.grad(res_keops, x_i, u)

        assert torch.allclose(res_keops_grad, res_torch_grad)

    def test_vmap_2(self):
        x_i = torch.randn(10, 5, 1, 2, requires_grad=test_grad, dtype=torch.float64)
        x_j = torch.randn(5, 1, 20, 2, dtype=torch.float64)
        y_j = torch.randn(1, 5, 20, 1, dtype=torch.float64)
        in_dims = (1, 0, 1)
        out_dims = 2

        res_keops = torch.vmap(fn, in_dims=in_dims, out_dims=out_dims)(
            x_i, x_j, y_j, use_keops=True
        )
        res_torch = torch.vmap(fn, in_dims=in_dims, out_dims=out_dims)(
            x_i, x_j, y_j, use_keops=False
        )
        assert torch.allclose(res_keops, res_torch)

        u = torch.rand(res_torch.shape)
        (res_torch_grad,) = torch.autograd.grad(res_torch, x_i, u)
        (res_keops_grad,) = torch.autograd.grad(res_keops, x_i, u)

        assert torch.allclose(res_keops_grad, res_torch_grad)
