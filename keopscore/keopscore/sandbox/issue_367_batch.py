import torch
from pykeops.torch import LazyTensor
from time import time


def fun_torch(A, I, J):
    return A[:, I, J].sum(axis=2)


def fun_keops(A, I, J):
    b, ncol, nrow = A.shape
    A = LazyTensor(A.reshape(b,1,1,ncol*nrow))
    I = LazyTensor(I.to(dtype)[..., None])
    J = LazyTensor(J.to(dtype)[..., None])
    K = A[I * ncol + J]
    return K.sum(axis=2).reshape(b,I.shape[1])


P, Q = 12, 5
B, M, N = 3, 1000, 1000
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
A = torch.randn((B, P, Q), requires_grad=True, device=device, dtype=dtype)
I = torch.randint(P, (B, M, 1), device=device)
J = torch.randint(Q, (B, 1, N), device=device)

test_torch = True

if test_torch:
    start = time()
    res_torch = fun_torch(A, I, J)
    end = time()
    print("time for torch:", end - start)
    # print(res_torch)

start = time()
res_keops = fun_keops(A, I, J)
end = time()
print("time for keops:", end - start)
# print(res_keops)

if test_torch:
    print(torch.norm(res_keops - res_torch) / torch.norm(res_torch))


# testing gradients
if test_torch:
    loss_torch = (res_torch**2).sum()
    res_torch = torch.autograd.grad(loss_torch, [A])[0]

loss_keops = (res_keops**2).sum()
res_keops = torch.autograd.grad(loss_keops, [A])[0]

if test_torch:
    print(torch.norm(res_keops - res_torch) / torch.norm(res_torch))

