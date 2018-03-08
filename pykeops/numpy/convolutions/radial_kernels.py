import numpy as np
import ctypes
from ctypes import POINTER, c_int, c_float

from pykeops.numpy.get_specific import get_specific_lib
from pykeops.common.compile_routines import compile_specific_routine

import os.path

# create __cuda_conv function with get_cuda_conv()
signature=[c_float, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int,c_int,c_int,c_int]
__radial_kernels_convs = get_specific_lib('radial_kernels_conv',signature)

# convenient python wrapper for __cuda_conv it does all job with types convertation from python ones to C++ ones 
def radial_kernels_conv(x, y, beta, result, sigma, kernel = "gaussian"):
    """
    Implements the operation :

    (x_i, y_j, beta_j)  ->  ( \sum_j k(x_i,y_j) beta_j )_i ,

    where k is a kernel function of parameter "sigma".
    Unlike a naive implementation, this code won't store in memory the matrix
    k(x_i,y_j) : it is therefore possible to use it when len(x) and len(y) are both large
    without getting a "memory overflow".

    N.B.: in an LDDMM setting, one would typically use "x = y = q", "beta = p". 
    """
    # From python to C float pointers and int :
    x_p      =      x.ctypes.data_as(POINTER(c_float))
    y_p      =      y.ctypes.data_as(POINTER(c_float))
    beta_p   =   beta.ctypes.data_as(POINTER(c_float))
    result_p = result.ctypes.data_as(POINTER(c_float))

    nx       =    x.shape[0]
    ny       =    y.shape[0]
    dimPoint =    x.shape[1]
    dimVect  = beta.shape[1]

    ooSigma2 = float(1/ (sigma*sigma)) # Compute this once and for all

    # Let's use our GPU, which works "in place" :
    __radial_kernels_convs[kernel](ooSigma2, x_p, y_p, beta_p, result_p, dimPoint, dimVect, nx, ny )



if __name__ == '__main__':
    """
    testing, benchmark convolution with two naive python implementations of the Gaussian convolution
    """
    np.set_printoptions(linewidth=200)

    sizeX    = int(600)
    sizeY    = int(100)
    dimPoint = int(3)
    dimVect  = int(3)
    sigma    = float(2)

    if True : # Random test
            x    = np.random.rand(sizeX,dimPoint).astype('float32')
            y    = np.random.rand(sizeY,dimPoint).astype('float32')
            beta = np.random.rand(sizeY,dimVect ).astype('float32')
    else :    # Deterministic one
            x    = np.sin([ np.arange(float(sizeX)), np.arange(float(sizeX))* 2,np.arange(float(sizeX))**2] ).astype('float32')
            y    = np.cos([ np.arange(float(sizeY)), np.arange(float(sizeY))* 2,np.arange(float(sizeY))**2] ).astype('float32')
            beta = np.array(np.arange(float(dimVect*sizeY)) +1).reshape(dimVect,sizeY).astype('float32')

    # Call cuda kernel
    gamma = np.zeros(dimVect*sizeX).astype('float32')
    radial_kernels_conv(x, y, beta, gamma, sigma) # In place, gamma_i = k(x_i,y_j) @ beta_j
    gamma = gamma.reshape((sizeX,dimVect))

    # A implementation using numpy
    oosigma2 = 1 / (sigma * sigma) 

    def squared_distances(x, y):
            return np.sum((x[:,np.newaxis,:] - y[np.newaxis,:,:]) ** 2, axis=2)

    def differences(x, y):
            return (x.T[:,:,np.newaxis] - y.T[:,np.newaxis,:])

    # A=exp(-|x_i - y_j|^2/(ker^2)).
    A = np.exp(-squared_distances(x, y) * oosigma2)
    gamma_py = (np.matmul(A,beta))

    # compare output
    print("\nCuda convolution :")
    print(gamma)

    print("\nPython convolution 1 :")
    print(gamma_py)

    print("\nIs everything okay ? ")
    print(np.allclose(gamma, gamma_py ))
