import torch
from pykeops.torch import LazyTensor

import keopscore

keopscore.auto_factorize = False

B1, B2, M, N, D = 5, 4, 10, 20, 2

x_i = torch.randn(B1, B2, M, 1, D)
y_j = torch.randn(B1, 1, 1, N, D)
b_j = torch.randn(B1, B2, 1, N, 1)
p = torch.rand(B1, 1, 1, 1, 1)
dx_i = torch.randn(B1, B2, M, 1, D)
dy_j = torch.randn(B1, 1, 1, N, D)
db_j = torch.randn(B1, B2, 1, N, 1)
dp = torch.randn(B1, 1, 1, 1, 1)


def fn_torch(x_i, y_j, b_j, p):
    K_ij = (-p[..., 0] * ((x_i - y_j) ** 2).sum(-1)).exp()
    K_ij = K_ij[..., None]
    return ((K_ij * b_j).sum(2)).norm()


def fn_keops(x_i, y_j, b_j, p):
    x_i = LazyTensor(x_i)
    y_j = LazyTensor(y_j)
    b_j = LazyTensor(b_j)
    p = LazyTensor(p)
    K_ij = (-p * ((x_i - y_j) ** 2).sum(-1)).exp()
    return ((K_ij * b_j).sum(2)).norm()


# 1) testing torch.func.vmap
res1 = torch.func.vmap(fn_torch)(x_i, y_j, b_j, p)
res2 = torch.func.vmap(fn_keops)(x_i, y_j, b_j, p)
print("testing vmap, error=", torch.norm(res1 - res2) / torch.norm(res1))

# 2) testing torch.func.grad
res1 = torch.func.grad(fn_torch, (0, 1, 2, 3))(x_i, y_j, b_j, p)
res2 = torch.func.grad(fn_keops, (0, 1, 2, 3))(x_i, y_j, b_j, p)
for k in range(4):
    print("testing grad, error=", torch.norm(res1[k] - res2[k]) / torch.norm(res1[k]))

# 3) testing torch.func.vjp
res1 = torch.func.vjp(fn_torch, x_i, y_j, b_j, p)[1](torch.tensor(1.0))[0]
res2 = torch.func.vjp(fn_keops, x_i, y_j, b_j, p)[1](torch.tensor(1.0))[0]
print("testing vjp, error=", torch.norm(res1 - res2) / torch.norm(res1))

# 4) testing torch.func.jvp
_, res1 = torch.func.jvp(fn_torch, (x_i, y_j, b_j, p), (dx_i, dy_j, db_j, dp))
_, res2 = torch.func.jvp(fn_keops, (x_i, y_j, b_j, p), (dx_i, dy_j, db_j, dp))
print("testing jvp, error=", torch.norm(res1 - res2) / torch.norm(res1))

# 5) testing torch.func.jacrev
res1 = torch.func.jacrev(fn_torch, (0, 1, 2, 3))(x_i, y_j, b_j, p)
res2 = torch.func.jacrev(fn_keops, (0, 1, 2, 3))(x_i, y_j, b_j, p)
for k in range(4):
    print("testing jacrev, error=", torch.norm(res1[k] - res2[k]) / torch.norm(res1[k]))

# 6) testing torch.func.jacfwd.
res1 = torch.func.jacfwd(fn_torch)(x_i, y_j, b_j, p)
res2 = torch.func.jacfwd(fn_keops)(x_i, y_j, b_j, p)
print("testing jacfwd, error=", torch.norm(res1 - res2) / torch.norm(res1))


# 7) testing torch.func.hessian.
res1 = torch.func.hessian(fn_torch)(x_i, y_j, b_j, p)
print("norm(res1)=", torch.norm(res1))
res2 = torch.func.hessian(fn_keops)(x_i, y_j, b_j, p)
print("norm(res2)=", torch.norm(res2))
print("testing hessian, error=", torch.norm(res1 - res2) / torch.norm(res1))
