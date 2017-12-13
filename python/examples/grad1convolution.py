import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..')

import time

from pykp.convolutions import cudagrad1conv
import numpy as np

N = 5000 ; M = 12000; D = 3; E = 3

a = np.random.rand(N,E).astype('float32')
x = np.random.rand(N,D).astype('float32')
y = np.random.rand(M,D).astype('float32')
b = np.random.rand(M,E).astype('float32')
s = np.array([2.4]).astype('float32')

def dgaussian(r2,s):
    return - np.exp(-r2/s**2) / (s **2)

def dlaplacian(r2,s):
    t = -np.sqrt(r2 + s**2)
    return np.exp(t) / (2*t)
    
def denergy(r2,s):
    return -.25 / (r2 + s**2) **(1.25)

def squdistance_matrix(ax,by):
    return np.sum( (x[:,np.newaxis,:] - y[np.newaxis,:,:]) **2, axis=2)
    
def chain_rules(q,ax,by,Aa,p):
    res = np.zeros(ax.shape).astype('float32')
    for i in range(ax.shape[1]):
        #Computation of 2*|x_i -x_j|*exp(-|x_i -x_j|^2/(lam^2))/(lam^2)
        ximyj = (np.tile(ax[:,i],[M,1]).T - np.tile(by[:,i],[N,1])) 
        res[:,i] = np.sum(q * ((2 * ximyj * Aa) @ p),axis=1)
    return res


#dry run to initialize GPU
g = np.zeros(x.shape).astype('float32')
cudagrad1conv.cuda_grad1conv(a, x, y, b, g, s)
#end of dry run

##############################
# Gaussian kernel
##############################
print("\n ---- Gaussian kernel  ")

start = time.perf_counter()
g = np.zeros(x.shape).astype('float32')
cudagrad1conv.cuda_grad1conv(a, x, y, b, g, s)
#print("Cuda:\n", g)
print("Time for cuda:   ", time.perf_counter() - start)

start = time.perf_counter()
A =  dgaussian(squdistance_matrix(x,y),s) 
g2 = chain_rules(a,x,y,A,b)
#print("Python:\n", g)
print("Time for python: ", time.perf_counter() - start)

print("absolute error: ", np.max(np.abs( (g - g2)))) 


##############################
# Laplace kernel
##############################
print("\n ---- Laplace kernel  ")

start = time.perf_counter()
g = np.zeros(x.shape).astype('float32')
cudagrad1conv.cuda_grad1conv(a, x, y, b, g, s, kernel = "laplacian")
print("Time for cuda:   ", time.perf_counter() - start)

start = time.perf_counter()
A =  dlaplacian(squdistance_matrix(x,y),s) 
g2 = chain_rules(a,x,y,A,b)
print("Time for python: ", time.perf_counter() - start)

print("Absolute error: ", np.max(np.abs( (g - g2))))


##############################
# Energy kernel
##############################
print("\n ---- Energy kernel  ")

start = time.perf_counter()
g = np.zeros(x.shape).astype('float32')
cudagrad1conv.cuda_grad1conv(a, x, y, b, g, s, kernel = "energy")
print("Time for cuda:   ", time.perf_counter() - start)

start = time.perf_counter()
A =  denergy(squdistance_matrix(x,y),s) 
g2 = chain_rules(a,x,y,A,b)
print("Time for python: ", time.perf_counter() - start)

print("Absolute error: ", np.max(np.abs( (g - g2))))


