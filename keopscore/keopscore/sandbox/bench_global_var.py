import torch
import pykeops
import keopscore
from pykeops.torch import LazyTensor
from time import time

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64


def fun_keops(A, I, J):
    ncol = A.shape[1]
    A = LazyTensor(A.flatten())
    I = LazyTensor(I.to(dtype)[..., None])
    J = LazyTensor(J.to(dtype)[..., None])
    K = A[I * ncol + J]
    return K.sum(axis=1).flatten()


def bench(Ptab, lim_dim_local_var):
    keopscore.config.config.lim_dim_local_var_i = lim_dim_local_var
    keopscore.config.config.lim_dim_local_var_j = lim_dim_local_var
    pykeops.clean_pykeops()
    M, N = 100000, 100000
    times = []
    for P in Ptab:
        try:
            A = torch.randn((P, P), requires_grad=True, device=device, dtype=dtype)
            I = torch.randint(P, (M, 1), device=device)
            J = torch.randint(P, (1, N), device=device)
            fun_keops(A, I[:100, :], J[:, :100])
            start = time()
            fun_keops(A, I, J)
            end = time()
            times.append(end - start)
        except:
            break
    return times


Ptab = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
lim_dims = [0, 10, 1e9]
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
for lim_dim in lim_dims:
    times = bench(Ptab, lim_dim)
    plt.loglog(Ptab[: len(times)], times)
plt.legend(["global", "local_global", "local"])
plt.savefig(
    "/home/glaunes/Bureau/keops/keopscore/keopscore/sandbox/output/bench_local_global.png"
)
