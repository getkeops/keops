import torch
from pykeops.symbolictensor.pytorch import keops

# testing symbolic tensors and keops decorator

# some data
M, N, D = 4, 3, 2
x = torch.rand(M, 1, D)
y = torch.rand(1, N, D)
b = torch.rand(1, N)


# a function, simpe gauss kernel
def gauss_kernel(x, y, b):
    dist2 = ((x - y) ** 2).sum(axis=2)
    K = (-dist2).exp()
    f = K * b
    return f.sum(axis=1)


out1 = gauss_kernel(x, y, b)
print(out1.shape)


# same function with the keops decorator
@keops
def gauss_kernel(x, y, b):
    dist2 = ((x - y) ** 2).sum(axis=2)
    K = (-dist2).exp()
    f = K * b
    return f.sum(axis=1)


out2 = gauss_kernel(x, y, b)
print(out2.shape)

print(torch.norm(out1 - out2))
