import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..')

import time

from pykp import cudaconv
import numpy as np

import torch

N = 9000 ; M = 12000; D = 3; E = 3

# declare numpy 
x = np.random.randn(N,D).astype('float32')
y = np.random.randn(M,D).astype('float32')
b = np.random.randn(M,E).astype('float32')
s = np.array([2.4]).astype('float32')

xc = torch.from_numpy(x).cuda()
yc = torch.from_numpy(y).cuda()
bc = torch.from_numpy(b).cuda()
sc = torch.from_numpy(s).cuda()

def np_kernel(x, y, s, kernel) :
    sq = np.sum( (x[:,np.newaxis,:] - y[np.newaxis,:,:]) **2, axis=2)
    if   kernel == "gaussian"  : return np.exp( -sq / (s*s))
    elif kernel == "laplacian" : return np.exp( -np.sqrt(sq + s*s))
    elif kernel == "energy"    : return 1. / ( s*s + sq ) **.25 

# declare the torch counterpart

def torch_kernel(x, y, s, kernel) :
    sq = torch.sum( (x[:,None]-y[None])**2 , 2 ) 
    if   kernel == "gaussian"  : return torch.exp( -sq / (s*s))
    elif kernel == "laplacian" : return torch.exp( -torch.sqrt(sq + s*s))
    elif kernel == "energy"    : return torch.pow( 1. / ( s*s + sq ), .25 )


# dry run to initialize GPU
# g = np.zeros([N,E]).astype('float32')
# cudaconv.cuda_conv(x, y, b, g, s)
# end of dry run

##############################
# Gaussian kernel
##############################


for k in (["gaussian", "laplacian", "energy"]):
    # cuda pytorch
    start = time.perf_counter()
    g0 = torch.mm(torch_kernel(xc,yc,sc,kernel=k),bc).cpu().numpy()
    print("\nTime for Pytorch/cuda: ", time.perf_counter() - start)

    # cuda tiled implementation
    start = time.perf_counter()
    g1 = np.zeros([N,E]).astype('float32')
    cudaconv.cuda_conv(x, y, b, g1, s, kernel=k)
    print("Time for cuda:   ", time.perf_counter() - start)

    # pure numpy
    start = time.perf_counter()
    g2 =  np_kernel(x,y,s,kernel=k) @ b
    print("Time for Python: ", time.perf_counter() - start)

    print("absolute error: ", np.max(np.abs (g1 - g0)))
