import torch
from pykeops.torch import LazyTensor


def fn_torch(x_i, x_j, y_j):
    K_ij = (-((x_i - x_j) ** 2).sum(-1)).exp()
    K_ij = K_ij[..., None]
    return ((K_ij * y_j).sum(1)).norm()


def fn_keops(x_i, x_j, y_j):
    x_i = LazyTensor(x_i)
    x_j = LazyTensor(x_j)
    y_j = LazyTensor(y_j)
    K_ij = (-((x_i - x_j) ** 2).sum(-1)).exp()
    return ((K_ij * y_j).sum(1)).norm()


x_i = torch.randn(5, 10, 1, 2)
x_j = torch.randn(5, 1, 20, 2)
y_j = torch.randn(5, 1, 20, 1)

# 1) testing torch.func.vmap
res1 = torch.func.vmap(fn_torch)(x_i, x_j, y_j)
res2 = torch.func.vmap(fn_keops)(x_i, x_j, y_j)
print("testing vmap, error=", torch.norm(res1 - res2) / torch.norm(res1))

# 2) testing torch.func.grad
res1 = torch.func.grad(fn_torch, (0, 1, 2))(x_i, x_j, y_j)
res2 = torch.func.grad(fn_keops, (0, 1, 2))(x_i, x_j, y_j)
for k in range(3):
    print("testing grad, error=", torch.norm(res1[k] - res2[k]) / torch.norm(res1[k]))

# 3) testing torch.func.vjp
res1 = torch.func.vjp(fn_torch, x_i, x_j, y_j)[1](torch.tensor(1.0))[0]
res2 = torch.func.vjp(fn_keops, x_i, x_j, y_j)[1](torch.tensor(1.0))[0]
print("testing vjp, error=", torch.norm(res1 - res2) / torch.norm(res1))

# 4) testing torch.func.jvp. Doesn't work because requires forward AD
# _, res1 = torch.func.jvp(fn_torch, (x_i, x_j, y_j), (x_i, x_j, y_j))
# _, res2 = torch.func.jvp(fn_keops, (x_i, x_j, y_j), (x_i, x_j, y_j))
# print("testing jvp, error=",torch.norm(res1 - res2) / torch.norm(res1))

# 5) testing torch.func.jacrev
res1 = torch.func.jacrev(fn_torch, (0, 1, 2))(x_i, x_j, y_j)
res2 = torch.func.jacrev(fn_keops, (0, 1, 2))(x_i, x_j, y_j)
for k in range(3):
    print("testing jacrev, error=", torch.norm(res1[k] - res2[k]) / torch.norm(res1[k]))

# 6) testing torch.func.jacfwd. Doesn't work because requires forward AD
# res1 = torch.func.jacfwd(fn_torch)(x_i, x_j, y_j)
# res2 = torch.func.jacfwd(fn_keops)(x_i, x_j, y_j)
# print("testing jacfwd, error=",torch.norm(res1 - res2) / torch.norm(res1))

# 7) testing torch.func.hessian. Doesn't work because requires forward AD
# res1 = torch.func.hessian(fn_torch)(x_i, x_j, y_j)
# res2 = torch.func.hessian(fn_keops)(x_i, x_j, y_j)
# print("testing hessian, error=",torch.norm(res1 - res2) / torch.norm(res1))
