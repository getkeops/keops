import torch
from pykeops.torch import LazyTensor

nptsx = 2
nptsy = 2
patch_size = 7 # OK for 6 with or without GPU, unstable for 7 with GPU

torch.manual_seed(0)
x=torch.rand(nptsx, 3*patch_size**2).double()
x.requires_grad = True
y=torch.rand(nptsy, 3*patch_size**2).double()

# compute matrix of square distances between patches
x_i = LazyTensor(x[:,None,:])
y_j = LazyTensor(y[None,:,:])
D_ij = ((x_i-y_j)**2).sum(2)
print(D_ij)
eps = 1e-2
S = (-1/eps*D_ij).logsumexp(1)
print("S: ", S)
xgrad = torch.autograd.grad(S,x,torch.ones_like(S))[0]
print(xgrad)