from pykeops.torch import LazyTensor, Vi, Vj
import torch

d = 3
x = Vi(0,d)
y = Vj(1,d)
D2 = x.sqdist(y)
K = (-D2).exp()*D2

print(K)

K2 = K.factorize(D2)

print(K2)

n = 100
x = torch.rand(n,d)
y = torch.rand(n,d)
b = torch.rand(n,d)

a = (K@b)(x,y)

a2 = (K2@b)(x,y)

print(a)

