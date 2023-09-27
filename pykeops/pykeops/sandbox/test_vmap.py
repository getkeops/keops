import torch
from pykeops.torch import LazyTensor

test_grad = True


def fn(x_i, x_j, y_j, use_keops=True):
    if use_keops:
        x_i = LazyTensor(x_i)
        x_j = LazyTensor(x_j)
        y_j = LazyTensor(y_j)
    K_ij = (-((x_i - x_j) ** 2).sum(-1)).exp()
    if not use_keops:
        K_ij = K_ij[..., None]
    return (K_ij * y_j).sum(1)


test_case = 1

if test_case == 0:
    x_i = torch.randn(5, 10, 1, 2, requires_grad=test_grad)
    x_j = torch.randn(5, 1, 20, 2)
    y_j = torch.randn(5, 1, 20, 1)
    in_dims = 0
    out_dims = 0
elif test_case == 1:
    x_i = torch.randn(10, 5, 1, 2, requires_grad=test_grad)
    x_j = torch.randn(1, 5, 20, 2)
    y_j = torch.randn(1, 5, 20, 1)
    in_dims = 1
    out_dims = 1
elif test_case == 2:
    x_i = torch.randn(10, 5, 1, 2, requires_grad=test_grad)
    x_j = torch.randn(5, 1, 20, 2)
    y_j = torch.randn(1, 5, 20, 1)
    in_dims = (1, 0, 1)
    out_dims = 2

# vmap with KeOps
res_keops = torch.vmap(fn, in_dims=in_dims, out_dims=out_dims)(
    x_i, x_j, y_j, use_keops=True
)

# vmap with torch
res_torch = torch.vmap(fn, in_dims=in_dims, out_dims=out_dims)(
    x_i, x_j, y_j, use_keops=False
)

print(torch.norm(res_keops - res_torch) / torch.norm(res_torch))

if test_grad:
    print("testing grad")
    u = torch.rand(res_torch.shape)
    (res_torch_grad,) = torch.autograd.grad(res_torch, x_i, u)
    (res_keops_grad,) = torch.autograd.grad(res_keops, x_i, u)
    print(torch.norm(res_keops_grad - res_torch_grad) / torch.norm(res_torch_grad))
