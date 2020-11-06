"""
=========
Multi GPU 
=========

On multi-device clusters,
let's see how to make use of several Gpus for further speedups

 
"""

###############################################################
# Setup
# -------------
# Standard imports:

import time
import torch
from pykeops.torch import LazyTensor
from concurrent.futures import ThreadPoolExecutor, wait

###############################################################
# Define the number of gpus

ngpus = torch.cuda.device_count()

###############################################################
#  Generate some data, stored on the CPU (host) memory:
#
M = 1000000
N = 1000000
x = torch.randn(M, 3)
y = torch.randn(N, 3)

###############################################################
#  Define a symbolic gaussian kernel 
#  reduction using LazyTensor syntax
#
def GaussKernelSum(x,y):
    x_i = LazyTensor( x[:,None,:] )     # x_i.shape = (M, 1, 3)
    y_j = LazyTensor( y[None,:,:] )     # y_j.shape = (1, N, 3)
    D_ij = ((x_i - y_j)**2).sum(dim=2)  # Symbolic (M,N,1) matrix of squared distances
    K_ij = (- D_ij).exp()               # Symbolic (M,N,1) Gaussian kernel matrix
    return K_ij.sum(dim=1)
    
#########################################
# Compute on GPUs in parallel, dividing
# y data into several parts
#

start = time.time()
pool = ThreadPoolExecutor(ngpus)
subsize_y = int(N/ngpus)
futures = []
for gpuid in range(ngpus):
    with torch.cuda.device(gpuid):
        ygpu = y[gpuid*subsize_y:(gpuid+1)*subsize_y,:].cuda()
        xgpu = x.cuda()
        futures.append(pool.submit(GaussKernelSum, xgpu, ygpu))
wait(futures)
out_multi = 0
for gpuid in range(ngpus):
    out_multi += futures[gpuid].result().cpu()
elapsed = time.time() - start
print('multi-Gpu : ',elapsed)

#########################################
# Compare with single Gpu computation
#
start = time.time()
out_single = GaussKernelSum(x,y)
elapsed = time.time() - start
print('single-Gpu : ',elapsed)

#########################################
# Check outputs are the same
#
print('relative error : ',(torch.norm(out_single-out_multi)/torch.norm(out_single)).item())

