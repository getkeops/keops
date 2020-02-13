# test for openmp support

import numpy as np
import torch
import time
from pykeops.torch import LazyTensor


print("test sans batchdim")

# data
N = 100000
x = torch.randn(N, 3)
y = torch.randn(N, 3)
b = torch.randn(N, 1)

# test with keops
start = time.time()
x_i = LazyTensor( x[:,None,:] )
y_j = LazyTensor( y[None,:,:] )
b_j = LazyTensor( b[None,:,:] )
D_ij = ((x_i - y_j)**2).sum(dim=2)
K_ij = (- D_ij).exp()
res_keops = (K_ij*b_j).sum(dim=1, backend='CPU').flatten()
print("time with KeOps : ", time.time()-start)

if N < 15000:
    # test with torch
    start = time.time()
    x_i = x[:,None,:]
    y_j = y[None,:,:]
    b_j = b[None,:,:]
    D_ij = ((x_i - y_j)**2).sum(dim=2,keepdims=True)
    K_ij = (- D_ij).exp()
    res_torch = (K_ij*b_j).sum(dim=1).flatten()
    print("time with PyTorch : ", time.time()-start)
    print("error : ",torch.norm(res_keops-res_torch)/torch.norm(res_torch))


if N < 15000:
    # test with numpy
    start = time.time()
    x_i = x.numpy()[:,None,:]
    y_j = y.numpy()[None,:,:]
    b_j = b.numpy()[None,:,:]
    D_ij = ((x_i - y_j)**2).sum(axis=2,keepdims=True)
    K_ij = np.exp(- D_ij)
    res_numpy = (K_ij*b_j).sum(axis=1).flatten()
    print("time with NumPy : ", time.time()-start)
    print("error : ",np.linalg.norm(res_keops.numpy()-res_numpy)/np.linalg.norm(res_numpy))


'''
print("test avec batchdim")

# data
B, N = 10, 4000
x = torch.randn(B,N, 3)
y = torch.randn(B,N, 3)
b = torch.randn(B,N, 1)

# test with keops
start = time.time()
x_i = LazyTensor( x[:,:,None,:] )
y_j = LazyTensor( y[:,None,:,:] )
b_j = LazyTensor( b[:,None,:,:] )
D_ij = ((x_i - y_j)**2).sum(dim=3)
K_ij = (- D_ij).exp()
res_keops = (K_ij*b_j).sum(dim=2, backend='CPU').flatten()
print("time with KeOps : ", time.time()-start)

if N < 5000:
    # test with torch
    start = time.time()
    x_i = x[:,:,None,:]
    y_j = y[:,None,:,:]
    b_j = b[:,None,:,:]
    D_ij = ((x_i - y_j)**2).sum(dim=3,keepdim=True)
    K_ij = (- D_ij).exp()
    res_torch = (K_ij*b_j).sum(dim=2).flatten()
    print("time with PyTorch : ", time.time()-start)
    print("error : ",torch.norm(res_keops-res_torch)/torch.norm(res_torch))
'''

