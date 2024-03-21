import torch
from pykeops.torch import LazyTensor, Genred

device = "cuda" if torch.cuda.is_available() else "cpu"

B, M, N = 3, 5, 4

x = LazyTensor(torch.rand(1, M, 1, 1, device=device))
y = LazyTensor(torch.rand(1, 1, N, 1, device=device))
p = LazyTensor(torch.rand(B, 1, 1, 1, device=device))

res = (-p*(x-y)**2).exp().sum(dim=2)

print(res.reshape((B,M)))


