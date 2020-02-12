# test for openmp support

import torch
import time
from pykeops.torch import LazyTensor


print("test sans batchdim")

# data
N = 4000
x = torch.randn(N, 3)
y = torch.randn(N, 3)

# test with keops
start = time.time()
x_i = LazyTensor( x[:,None,:] )
y_j = LazyTensor( y[None,:,:] )
D_ij = ((x_i - y_j)**2).sum(dim=2)
K_ij = (- D_ij).exp()
res_keops = K_ij.sum(dim=1).flatten()
print("time with KeOps : ", time.time()-start)

if N < 5000:
    # test with torch
    start = time.time()
    x_i = x[:,None,:]
    y_j = y[None,:,:]
    D_ij = ((x_i - y_j)**2).sum(dim=2)
    K_ij = (- D_ij).exp()
    res_torch = K_ij.sum(dim=1)
    print("time with PyTorch : ", time.time()-start)
    print("error : ",torch.norm(res_keops-res_torch)/torch.norm(res_torch))



print("test avec batchdim")

# data
B, N = 10, 4000
x = torch.randn(B,N, 3)
y = torch.randn(B,N, 3)

# test with keops
start = time.time()
x_i = LazyTensor( x[:,:,None,:] )
y_j = LazyTensor( y[:,None,:,:] )
D_ij = ((x_i - y_j)**2).sum(dim=3)
K_ij = (- D_ij).exp()
res_keops = K_ij.sum(dim=2).flatten()
print("time with KeOps : ", time.time()-start)

if N < 5000:
    # test with torch
    start = time.time()
    x_i = x[:,:,None,:]
    y_j = y[:,None,:,:]
    D_ij = ((x_i - y_j)**2).sum(dim=3)
    K_ij = (- D_ij).exp()
    res_torch = K_ij.sum(dim=2).flatten()
    print("time with PyTorch : ", time.time()-start)
    print("error : ",torch.norm(res_keops-res_torch)/torch.norm(res_torch))


