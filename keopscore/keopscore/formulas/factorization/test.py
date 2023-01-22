from pykeops.torch import LazyTensor, Vi, Vj
import torch
from time import time

d = 3
x = Vi(0, d)
y = Vj(1, d)
D2 = x.sqdist(y)
K = (-D2).exp() * D2

K2 = K.factorize(D2)

n = 10000
x = torch.rand(n, d)
y = torch.rand(n, d)
b = torch.rand(n, d)

a = (K @ b)(x, y)
start = time()
a = (K @ b)(x, y)
end = time()
print("time for a:", end - start)

a2 = (K2 @ b)(x, y)
start = time()
a2 = (K2 @ b)(x, y)
end = time()
print("time for a2:", end - start)

print("relative error", torch.norm(a - a2) / torch.norm(a))
