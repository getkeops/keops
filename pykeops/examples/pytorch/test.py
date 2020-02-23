

import time

import torch
from matplotlib import pyplot as plt

from pykeops.torch import Genred

###############################################################################
# Define our dataset:
#

M = 5000  # Number of "i" points
N = 5000  # Number of "j" points
D = 3    # Dimension of the ambient space
Dv = 2   # Dimension of the vectors

x = 2*torch.randn(M,D)
y = 2*torch.randn(N,D)
b = torch.rand(N,Dv)


formula = 'Sum(Sin(x-y))*b'
aliases = ['x = Vi('+str(D)+')',   # First arg:  i-variable of size D
           'y = Vj('+str(D)+')',   # Second arg: j-variable of size D
           'b = Vj('+str(Dv)+')']  # Third arg:  j-variable of size Dv

op = Genred(formula, aliases, reduction_op='Min', axis=1)

# Dummy first call to warmup the GPU and get accurate timings:
_ = op(x, y, b)

###############################################################################
# Use our new function on arbitrary Numpy arrays:
#

start = time.time()
c = op(x, y, b)
print("Timing (KeOps implementation): ",round(time.time()-start,5),"s")

# compare with direct implementation
start = time.time()
cc = torch.sum( torch.sin( x[:,None,:] - y[None,:,:] ) , 2)
cc = torch.min(cc[:,:,None]*b[None,:,:],dim=1)
cc = cc[0]
print("Timing (PyTorch implementation): ", round(time.time()-start,5),"s")

print("Relative error : ", (torch.norm(c - cc) / torch.norm(c)).item())
