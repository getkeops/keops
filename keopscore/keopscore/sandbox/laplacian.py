from pykeops.torch import Vi, Vj
import torch
from time import time
from keopscore.formulas import *

# import keopscore
# keopscore.debug_ops = True


def GaussLapKernel(sigma, D):
    x, y = Vi(0, D), Vj(1, D)
    D2 = x.sqdist(y)
    K = (-D2 / (2 * sigma**2)).exp()
    res = (K * (D2 - D * sigma**2) / sigma**4).sum_reduction(axis=1)
    return res


def GaussK(sigma):
    def K(z):
        return (-(z**2).sum(-1) / (2 * sigma**2)).exp()

    return K


def LapKernel(K, D):
    x, y = Vi(0, D), Vj(1, D)
    K1 = K(x - y).grad(x, 1)
    Klap = K1.elem(0).grad(x, 1).elem(0)
    for i in range(1, D):
        Klap = Klap + K1.elem(i).grad(x, 1).elem(i)
    res = Klap.sum_reduction(axis=1)
    return res


def LapKernel_alt(K, D):
    x, y = Vi(0, D), Vj(1, D)
    K1 = K(x - y).grad(x, 1)
    GK1 = K1.grad_matrix(x)
    Klap = GK1.elem(0)
    for i in range(1, D**2, D):
        Klap = Klap + GK1.elem(i)
    res = Klap.sum_reduction(axis=1)
    return res


def LapKernel_trace(K, D):
    x, y = Vi(0, D), Vj(1, D)
    u = Vi(2, D)
    K1 = K(x - y).grad(x, 1)
    GK1 = K1.grad(x, u)
    Klap = GK1.trace_operator(u)
    res = Klap.sum_reduction(axis=1)
    return res


def LapKernel_lap(K, D):
    x, y = Vi(0, D), Vj(1, D)
    return K(x - y).laplacian(x).sum_reduction(axis=1)


def LapKernel_lap_fact(K, D):
    x, y = Vi(0, D), Vj(1, D)
    return K(x - y).laplacian(x).auto_factorize().sum_reduction(axis=1)


sigma, D = 1.5, 3

f1 = GaussLapKernel(sigma, D)
f2 = LapKernel(GaussK(sigma), D)
f3 = LapKernel_trace(GaussK(sigma), D)
f4 = LapKernel_lap(GaussK(sigma), D)
f5 = LapKernel_lap_fact(GaussK(sigma), D)

M, N = (150000, 100000) if torch.cuda.is_available() else (15000, 10000)

x = torch.rand(M, D, requires_grad=True)
y = torch.rand(N, D)

u1 = f1(x, y)  # warming up GPU...
torch.cuda.synchronize()

start = time()
u1 = f1(x, y)
torch.cuda.synchronize()
end = time()
print("f1 : ", f1.formula)
print("time for u1:", end - start)

start = time()
u2 = f2(x, y)
torch.cuda.synchronize()
end = time()

print("f2 : ", f2.formula)
print("time for u2:", end - start)
print("error:", torch.norm(u1 - u2) / torch.norm(u1))

start = time()
u3 = f3(x, y)
torch.cuda.synchronize()
end = time()
print("f3 : ", f3.formula)
print("time for u3:", end - start)
print("error:", torch.norm(u1 - u3) / torch.norm(u1))

start = time()
u4 = f4(x, y)
torch.cuda.synchronize()
end = time()
print("f4 : ", f4.formula)
print("time for u4:", end - start)
print("error:", torch.norm(u1 - u4) / torch.norm(u1))

start = time()
u5 = f5(x, y)
torch.cuda.synchronize()
end = time()
print("f5 : ", f5.formula)
print("time for u5:", end - start)
print("error:", torch.norm(u1 - u5) / torch.norm(u1))


v1 = torch.norm(u1)
v2 = torch.norm(u2)
v3 = torch.norm(u3)
v4 = torch.norm(u4)
v5 = torch.norm(u5)
torch.cuda.synchronize()

start = time()
g1 = torch.autograd.grad(v1, x)[0]
torch.cuda.synchronize()
end = time()
print("time for g1:", end - start)

start = time()
g2 = torch.autograd.grad(v2, x)[0]
torch.cuda.synchronize()
end = time()
print("time for g2:", end - start)
print("error:", torch.norm(g1 - g2) / torch.norm(g1))

start = time()
g3 = torch.autograd.grad(v3, x)[0]
torch.cuda.synchronize()
end = time()
print("time for g3:", end - start)
print("error:", torch.norm(g1 - g3) / torch.norm(g1))

start = time()
g4 = torch.autograd.grad(v4, x)[0]
torch.cuda.synchronize()
end = time()
print("time for g4:", end - start)
print("error:", torch.norm(g1 - g4) / torch.norm(g1))

start = time()
g5 = torch.autograd.grad(v5, x)[0]
torch.cuda.synchronize()
end = time()
print("time for g5:", end - start)
print("error:", torch.norm(g1 - g5) / torch.norm(g1))
