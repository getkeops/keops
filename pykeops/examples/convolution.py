import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..')

import time, timeit

from pykp.convolutions import cudaconv
import numpy as np

import torch

N = 2000 ; M = 3000; D = 3; E = 3

# declare numpy 
x = np.random.randn(N,D).astype('float32')
y = np.random.randn(M,D).astype('float32')
b = np.random.randn(M,E).astype('float32')
s = np.array([2.4]).astype('float32')

xc = torch.from_numpy(x.copy()).cuda()
yc = torch.from_numpy(y.copy()).cuda()
bc = torch.from_numpy(b.copy()).cuda()
sc = torch.from_numpy(s.copy()).cuda()

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

enable_GC = False # Garbage collection?
GC = 'gc.enable();' if enable_GC else 'pass;'
LOOPS = 200
print("Time to compute ", LOOPS, " convolutions of size {}x{}:".format(N,M))
for k in (["gaussian", "laplacian", "energy"]):
    print(k, " kernel:")
    
    # For a 100% fair assessment, we make a dry run first.
    # cuda pytorch
    g0 = torch.mm(torch_kernel(xc,yc,sc,kernel=k),bc)#.cpu().numpy()
    T = time.time() ; print(T)
    speed_pytorch = timeit.Timer('g0 = torch.mm(torch_kernel(xc,yc,sc,kernel=k),bc)#.cpu().numpy()', 
                                 GC, globals = globals(),
                                 timer = time.time).timeit(LOOPS)
    T = time.time() - T
    print("Time for Pytorch/cuda: {:.4f}s, {:.4f}s".format(speed_pytorch, T))
    
    
    # cuda tiled implementation
    g1 = np.zeros([N,E]).astype('float32') ; cudaconv.cuda_conv(x, y, b, g1, s, kernel=k)
    g1 = np.zeros([N,E]).astype('float32')
    T = time.time() ; print(T)
    speed_libkp = timeit.Timer('cudaconv.cuda_conv(x, y, b, g1, s, kernel=k)', 
                               GC, globals = globals(),
                                timer = time.time).timeit(LOOPS)
    T = time.time() - T
    print("Time for cuda:         {:.4f}s, {:.4f}s".format(speed_libkp, T))
    
    if False :
        # pure numpy
        g2 =  np_kernel(x,y,s,kernel=k) @ b
        speed_numpy = timeit.Timer('g2 =  np_kernel(x,y,s,kernel=k) @ b', 
                                   GC, globals = globals(),
                                   timer = time.time).timeit(LOOPS)
        print("Time for Python:       {:.4f}s".format(speed_numpy))
        print("Absolute error:       ", np.max(np.abs (g1 - g2)), "\n")















