import torch
from pykeops.torch import LazyTensor

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
    # return ((D2_ij.exp() * b_j).sum(2)).log().norm()


# 1) testing torch.func.vmap
res1 = torch.func.vmap(fn_torch)(x_i, y_j, b_j)
res2 = torch.func.vmap(fn_keops)(x_i, y_j, b_j)
print("testing vmap, error=", torch.norm(res1 - res2) / torch.norm(res1))

# 2) testing torch.func.grad
res1 = torch.func.grad(fn_torch, (0, 1, 2))(x_i, y_j, b_j)
res2 = torch.func.grad(fn_keops, (0, 1, 2))(x_i, y_j, b_j)
for k in range(3):
    print("testing grad, error=", torch.norm(res1[k] - res2[k]) / torch.norm(res1[k]))

# 3) testing torch.func.vjp
res1 = torch.func.vjp(fn_torch, x_i, y_j, b_j)[1](torch.tensor(1.0))[0]
res2 = torch.func.vjp(fn_keops, x_i, y_j, b_j)[1](torch.tensor(1.0))[0]
print("testing vjp, error=", torch.norm(res1 - res2) / torch.norm(res1))

# 4) testing torch.func.jvp
_, res1 = torch.func.jvp(fn_torch, (x_i, y_j, b_j), (dx_i, dy_j, db_j))
_, res2 = torch.func.jvp(fn_keops, (x_i, y_j, b_j), (dx_i, dy_j, db_j))
print("testing jvp, error=", torch.norm(res1 - res2) / torch.norm(res1))

# 5) testing torch.func.jacrev
res1 = torch.func.jacrev(fn_torch, (0, 1, 2))(x_i, y_j, b_j)
res2 = torch.func.jacrev(fn_keops, (0, 1, 2))(x_i, y_j, b_j)
for k in range(3):
    print("testing jacrev, error=", torch.norm(res1[k] - res2[k]) / torch.norm(res1[k]))

# 6) testing torch.func.jacfwd.
res1 = torch.func.jacfwd(fn_torch)(x_i, y_j, b_j)
res2 = torch.func.jacfwd(fn_keops)(x_i, y_j, b_j)
print("testing jacfwd, error=", torch.norm(res1 - res2) / torch.norm(res1))

# 7) testing torch.func.hessian.
res1 = torch.func.hessian(fn_torch)(x_i, y_j, b_j)
res2 = torch.func.hessian(fn_keops)(x_i, y_j, b_j)
print("testing hessian, error=", torch.norm(res1 - res2) / torch.norm(res1))
