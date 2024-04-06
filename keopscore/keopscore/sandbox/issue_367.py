import torch
from pykeops.torch import LazyTensor
from time import time


def fun_torch(A, I, J):
    return A[I, J].sum(axis=1)


def fun_keops(A, I, J):
    ncol = A.shape[1]
    A = LazyTensor(A.flatten())
    I = LazyTensor(I.to(dtype)[..., None])
    J = LazyTensor(J.to(dtype)[..., None])
    K = A[I * ncol + J]
    return K.sum(axis=1).flatten()


P, Q = 10000, 10000
M, N = 100000, 100000
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float
A = torch.randn((P, Q), requires_grad=True, device=device, dtype=dtype)
I = torch.randint(P, (M, 1), device=device)
J = torch.randint(Q, (1, N), device=device)

test_torch = False

if test_torch:
    start = time()
    res_torch = fun_torch(A, I, J)
    end = time()
    print("time for torch:", end-start)
    #print(res_torch)

start = time()
res_keops = fun_keops(A, I, J)
end = time()
print("time for keops:", end-start)
#print(res_keops)

if test_torch:
    print(torch.norm(res_keops-res_torch)/torch.norm(res_torch))

"""
# testing gradients
loss_torch = (res_torch**2).sum()
print(torch.autograd.grad(loss_torch, [A]))

loss_keops = (res_keops**2).sum()
print(torch.autograd.grad(loss_keops, [A]))
"""
