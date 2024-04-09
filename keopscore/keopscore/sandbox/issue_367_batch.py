import torch
from pykeops.torch import LazyTensor
from time import time


def fun_torch(A, I, J):
    return A[:, I, J].diagonal().permute(2,0,1).sum(axis=2)


def fun_keops(A, I, J):
    b, nrow, ncol = A.shape
    A = LazyTensor(A.reshape(b,1,1,ncol*nrow))
    I = LazyTensor(I.to(dtype)[:,:,:, None])
    J = LazyTensor(J.to(dtype)[:,:,:, None])
    K = A[I * ncol + J]
    return K.sum(axis=2).reshape(I.shape[:2])


P, Q = 12, 5
B, M, N = 3, 10, 10
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
A = torch.randn((B, P, Q), requires_grad=True, device=device, dtype=dtype)
I = torch.randint(P, (B, M, 1), device=device)
J = torch.randint(Q, (B, 1, N), device=device)

test_torch = True
test_grad = False

if test_torch:
    start = time()
    res_torch = fun_torch(A, I, J)
    end = time()
    print("time for torch:", end - start)

start = time()
res_keops = fun_keops(A, I, J)
end = time()
print("time for keops:", end - start)

if test_torch:
    print(torch.norm(res_keops - res_torch) / torch.norm(res_torch))

if test_grad:
    # testing gradients
    if test_torch:
        loss_torch = (res_torch**2).sum()
        res_torch = torch.autograd.grad(loss_torch, [A])[0]

    loss_keops = (res_keops**2).sum()
    res_keops = torch.autograd.grad(loss_keops, [A])[0]

    if test_torch:
        print(torch.norm(res_keops - res_torch) / torch.norm(res_torch))

