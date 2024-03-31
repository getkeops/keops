import torch
from pykeops.torch import LazyTensor


def fun_torch(A, I, J):
    return A[I, J].sum(axis=1)


def fun_keops(A, I, J):
    ncol = A.shape[1]
    A = LazyTensor(A.flatten())
    I = LazyTensor((I + 0.0)[..., None])
    J = LazyTensor((J + 0.0)[..., None])
    K = A[I * ncol + J]
    return K.sum(axis=1).flatten()


P, Q = 12, 5
M, N = 3000, 2000
device = "cuda" if torch.cuda.is_available() else "cpu"
A = torch.randn((P, Q), requires_grad=True, device=device)
I = torch.randint(P, (M, 1), device=device)
J = torch.randint(Q, (1, N), device=device)

res_torch = fun_torch(A, I, J)
print(res_torch)

res_keops = fun_keops(A, I, J)
print(res_keops)

# testing gradients
loss_torch = (res_torch**2).sum()
print(torch.autograd.grad(loss_torch, [A]))

loss_keops = (res_keops**2).sum()
print(torch.autograd.grad(loss_keops, [A]))