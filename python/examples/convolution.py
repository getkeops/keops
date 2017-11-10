import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..')

import time

from pypk import cudaconv
import numpy as np

N = 5000 ; M = 12000; D = 3; E = 3

x = np.random.randn(N,D).astype('float32')
y = np.random.randn(M,D).astype('float32')
b = np.random.randn(M,E).astype('float32')
s = np.array([2.4]).astype('float32')

def gaussian(r2,s):
    return np.exp(-r2/s**2)

def laplacian(r2,s):
    return np.exp(-np.sqrt(r2 + s**2))
    
def energy(r2,s):
    return (r2 + s**2) **(-.25)

def squdistance_matrix(ax,by):
    return np.sum( (x[:,np.newaxis,:] - y[np.newaxis,:,:]) **2, axis=2)


#dry run to initialize GPU
g = np.zeros([N,E]).astype('float32')
cudaconv.cuda_conv(x, y, b, g, s)
#end of dry run

##############################
# Gaussian kernel
##############################
print("\n ---- Gaussian kernel  ")

start = time.perf_counter()
g = np.zeros([N,E]).astype('float32')
cudaconv.cuda_conv(x, y, b, g, s)
#print("cuda gaussianconv : \n", g)
print("Time for cuda:   ", time.perf_counter() - start)

start = time.perf_counter()
g2 =  gaussian(squdistance_matrix(x,y),s) @ b
#print("python gaussianconv : \n", g2)
print("Time for Python: ", time.perf_counter() - start)

print("absolute error: ", np.max(np.abs (g - g2)))


##############################
# Laplace kernel
##############################
print("\n ---- Laplace kernel  ")

start = time.perf_counter()
g = np.zeros([N,E]).astype('float32')
cudaconv.cuda_conv(x, y, b, g, s, kernel = "laplacian")
#print("cuda laplacianconv : \n", g)
print("Time for cuda:   ", time.perf_counter() - start)

start = time.perf_counter()
g2 =  laplacian(squdistance_matrix(x,y),s) @ b
#print("python laplacianianconv : \n", g2)
print("Time for Python: ", time.perf_counter() - start)

print("absolute error: ", np.max(np.abs (g - g2)))


##############################
# Energy kernel
##############################
print("\n ---- Energy kernel  ")

start = time.perf_counter()
g = np.zeros([N,E]).astype('float32')
cudaconv.cuda_conv(x, y, b, g, s, kernel = "energy")
#print("cuda energyconv : \n", g)
print("Time for cuda:   ", time.perf_counter() - start)

start = time.perf_counter()
g2 =  energy(squdistance_matrix(x,y),s) @ b
#print("python energyconv : \n", g2)
print("Time for Python: ", time.perf_counter() - start)

print("absolute error: ", np.max(np.abs (g - g2)))

