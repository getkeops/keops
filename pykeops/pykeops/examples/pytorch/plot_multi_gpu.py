"""
=============
GPU Selection
=============

On multi-device clusters,
let's see how to make use of several Gpus for further speedups

"""

###############################################################
# Setup
# -------------
# Standard imports:

import math
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
def GaussKernelSum(x, y, gpuid=-1):
    x_i = LazyTensor(x[:, None, :])  # x_i.shape = (M, 1, 3)
    y_j = LazyTensor(y[None, :, :])  # y_j.shape = (1, N, 3)
    D_ij = ((x_i - y_j) ** 2).sum(dim=2)  # Symbolic (M,N,1) matrix of squared distances
    K_ij = (-D_ij).exp()  # Symbolic (M,N,1) Gaussian kernel matrix
    return K_ij.sum(dim=1, device_id=gpuid)


###############################################################
#  dummy calls to the routine
#
for gpuid in range(ngpus):
    GaussKernelSum(x[:100, :], y[:100, :], gpuid)

###############################################################
# Compute on several GPUs in parallel,
# (block parallelizing over the reduction index)
#

start = time.time()
pool = ThreadPoolExecutor(ngpus)
subsize_y = math.ceil(N / ngpus)
futures = []
for gpuid in range(ngpus):
    y_chunk = y[gpuid * subsize_y : (gpuid + 1) * subsize_y, :]
    future = pool.submit(GaussKernelSum, x, y_chunk, gpuid)
    futures.append(future)
wait(futures)
out_multi_1 = 0
for gpuid in range(ngpus):
    out_multi_1 += futures[gpuid].result().cpu()
elapsed = time.time() - start
print(
    "time for multi-Gpu computation (block parallelized over reduction index):{:.2f} s".format(
        elapsed
    )
)

###############################################################
# Compute on several GPUs in parallel,
# (block parallelizing over the output index)
#

start = time.time()
pool = ThreadPoolExecutor(ngpus)
subsize_y = math.ceil(N / ngpus)
futures = []
for gpuid in range(ngpus):
    x_chunk = x[gpuid * subsize_y : (gpuid + 1) * subsize_y, :]
    future = pool.submit(GaussKernelSum, x_chunk, y, gpuid)
    futures.append(future)
wait(futures)
out_multi_2 = torch.zeros(M, 3)
for gpuid in range(ngpus):
    out_multi_2[gpuid * subsize_y : (gpuid + 1) * subsize_y, :] = (
        futures[gpuid].result().cpu()
    )
elapsed = time.time() - start
print(
    "time for multi-Gpu computation (block parallelized over output index):{:.2f} s".format(
        elapsed
    )
)

#########################################
# Compare with single Gpu computation
#
start = time.time()
out_single = GaussKernelSum(x, y)
elapsed = time.time() - start
print("time for single-Gpu computation: {:.2f} s".format(elapsed))

#########################################
# Check outputs are the same
#
rel_err1 = (torch.norm(out_single - out_multi_1) / torch.norm(out_single)).item()
rel_err2 = (torch.norm(out_single - out_multi_2) / torch.norm(out_single)).item()
print("relative errors: {:.1e}, {:.1e}".format(rel_err1, rel_err2))
