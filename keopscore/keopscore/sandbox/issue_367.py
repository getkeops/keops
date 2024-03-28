import torch
from pykeops.torch import LazyTensor


def fun_torch(A, I, J):
    return A[I, J].sum(axis=1)


def fun_keops(A, I, J):
    ncol = A.shape[1]
    A = LazyTensor(A.flatten()[None, None, :])
    I = LazyTensor((I + 0.0).flatten()[:, None, None])
    J = LazyTensor((J + 0.0).flatten()[None, :, None])
    K = A[I * ncol + J]
    return K.sum(axis=1).flatten()


A = torch.tensor([[3.0, 4, 5], [9, 6, 7]], requires_grad=True)
I = torch.tensor([[0, 1, 0, 1, 1]]).t()
J = torch.tensor([[2, 1, 0, 0]])

res_torch = fun_torch(A, I, J)
print(res_torch)

res_keops = fun_keops(A, I, J)
print(res_keops)

# testing gradients
loss_torch = (res_torch**2).sum()
print(torch.autograd.grad(loss_torch, [A]))

loss_keops = (res_keops**2).sum()
print(torch.autograd.grad(loss_keops, [A]))
