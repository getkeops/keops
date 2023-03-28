import torch

from pykeops.torch import LazyTensor
#LazyTensor = lambda x:x

N, M, C = 5000, 2000, 10  
a = LazyTensor(torch.randn(1, N, C))  
b =  LazyTensor(torch.randn(M, 1, C))
c = ((a-b)**2).sum(axis=2) 
print(c.shape) # this gives [M, N]
d = c[:,10:20]
print(d.shape) # this gives an error, but I hope to get the shape [M, 10]

