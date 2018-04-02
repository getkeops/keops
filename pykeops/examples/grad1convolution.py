import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import numpy as np
from pykeops.numpy.convolutions.radial_kernels_grad1 import radial_kernels_grad1conv

import time, timeit

N = 500 ; M = 300; D = 3; E = 3

# declare numpy 
a = np.random.rand(N,E).astype('float32')
x = np.random.rand(N,D).astype('float32')
y = np.random.rand(M,D).astype('float32')
b = np.random.rand(M,E).astype('float32')
s = np.array([2.4]).astype('float32')

def grad_np_kernel(x, y, s, kernel) :
    sq = np.sum( (x[:,np.newaxis,:] - y[np.newaxis,:,:]) **2, axis=2)
    if   kernel == "gaussian"  : return - np.exp(-sq/s*s) / (s*s)
    elif kernel == "laplacian" : t = -np.sqrt(sq / (s*s)) ; return  np.exp(t) / (2*s*s*t)
    elif kernel == "cauchy"    : return -1. / (s * (sq/(s*s) + 1) )**2 
    elif kernel == "inverse_multiquadric"    : return -.5 / (sq + s**2) **1.5 

    
def chain_rules(q,ax,by,Aa,p):
    res = np.zeros(ax.shape).astype('float32')
    for i in range(ax.shape[1]):
        #Computation of 2*|x_i -x_j|*exp(-|x_i -x_j|^2/(lam^2))/(lam^2)
        ximyj = (np.tile(ax[:,i],[M,1]).T - np.tile(by[:,i],[N,1])) 
        res[:,i] = np.sum(q * ((2 * ximyj * Aa) @ p),axis=1)
    return res


##############################
# Benchmark
##############################

enable_GC = False # Garbage collection?
GC = 'gc.enable();' if enable_GC else 'pass;'
LOOPS = 200
print("Time to compute ", LOOPS, " convolutions of size {}x{}:".format(N,M))
print("\n",end="")

for k in (["gaussian", "laplacian", "cauchy", "inverse_multiquadric"]):
    print(k, " kernel:")
    # cuda tiled implementation
    g1 = np.zeros([N,E]).astype('float32') ; radial_kernels_grad1conv(a, x, y, b, g1, s, kernel=k)
    g1 = np.zeros([N,E]).astype('float32')
    speed_pykeops = timeit.Timer('radial_kernels_grad1conv(a, x, y, b, g1, s, kernel=k)', GC, globals = globals(), timer = time.time).timeit(LOOPS)
    print("Time for cuda:         {:.4f}s".format(speed_pykeops))


    # pure numpy
    g2 = chain_rules(a,x,y,grad_np_kernel(x,y,s,kernel=k),b)

    speed_numpy = timeit.Timer('g2 = chain_rules(a,x,y,grad_np_kernel(x,y,s,kernel=k),b)', GC, globals = globals(), timer = time.time).timeit(LOOPS)
    print("Time for Python:       {:.4f}s".format(speed_numpy))
    print("Absolute error:       ", np.max(np.abs (g1 - g2)), "\n")
