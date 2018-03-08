import numpy as np
import ctypes
from ctypes import POINTER, c_int, c_float

from pykeops.numpy.get_specific import get_specific_lib
from pykeops.common.compile_routines import compile_specific_routine

import os.path


# create __cuda_grad1conv function with get_cuda_grad1conv()
signature=[     c_float, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int,    c_int,   c_int, c_int  ]

__radial_kernels_grad1convs = get_specific_lib('radial_kernels_grad1conv',signature)

# convenient python wrapper for __cuda_grad1conv it does all job with types convertation from python ones to C++ ones 
def radial_kernels_grad1conv(alpha,x, y, beta, result, sigma, kernel = "gaussian"):
    """
    Implements the operation :

    (alpha_i, x_i, y_j, beta_j)  ->  (\partial_{x_i} < alpha_i | ( \sum_j k(x_i,y_j) beta_j )_i >)_i ,

    where k is a kernel function of parameter "sigma".
    Unlike a naive pytorch implementation, this code won't store in memory the matrix
    k(x_i,y_j) : it is therefore possible to use it when len(x) and len(y) are both large
    without getting a "memory overflow".

    N.B.: in an LDDMM setting, one would typically use "x = y = q", "beta = p". 
    """
    # From python to C float pointers and int :
    alpha_p  =  alpha.ctypes.data_as(POINTER(c_float))
    x_p      =      x.ctypes.data_as(POINTER(c_float))
    y_p      =      y.ctypes.data_as(POINTER(c_float))
    beta_p   =   beta.ctypes.data_as(POINTER(c_float))
    result_p = result.ctypes.data_as(POINTER(c_float))

    nx       =    x.shape[0]
    ny       =    y.shape[0]
    dimPoint =    x.shape[1]
    dimVect  = beta.shape[1]

    ooSigma2 = float(1/ (sigma*sigma))  # Compute this once and for all

    # Let's use our GPU, which works "in place" :
    __radial_kernels_grad1convs[kernel](ooSigma2, alpha_p, x_p, y_p, beta_p, result_p, dimPoint, dimVect, nx, ny )





# testing, benchmark grad1convolution with a naive python implementation of the Gaussian convolution
if __name__ == '__main__':

    np.set_printoptions(linewidth=200)
    sizeX    = int(10)
    sizeY    = int(11)
    dimPoint = int(3)
    dimVect  = int(3)
    sigma    = float(2)

    alpha = np.random.rand(sizeX,dimVect ).astype('float32')
    x     = np.random.rand(sizeX,dimPoint).astype('float32')
    y     = np.random.rand(sizeY,dimPoint).astype('float32')
    beta  = np.random.rand(sizeY,dimVect ).astype('float32')

    # Call cuda kernel
    gamma = np.zeros(dimPoint*sizeX).astype('float32')
    cuda_grad1conv(alpha,x, y, beta, gamma, sigma) # In place, gamma_i = k(x_i,y_j) @ beta_j
    gamma = gamma.reshape((sizeX,dimPoint))

    # A first implementation
    oosigma2 = 1 / (sigma * sigma) 

    def squared_distances(x, y):
        return np.sum((x[:,np.newaxis,:] - y[np.newaxis,:,:]) ** 2, axis=2)

    def differences(x, y):
        return (x.T[:,:,np.newaxis] - y.T[:,np.newaxis,:])

    # A=exp(-|x_i - y_j|^2/(ker^2)).
    A = np.exp(-squared_distances(x, y) * oosigma2)

    # B=2*(x_i - y_j)*exp(-|x_i - y_j|^2/(ker^2))/(ker^2).
    B = (differences(x, y) * A)

    gamma_py = -2*(np.sum( alpha * (np.matmul(B,beta)),axis=2) * oosigma2).T

    # A second one using torch:
    """
    import torch

    x = torch.Tensor(x)
    y = torch.Tensor(y)
    pp = torch.Tensor(alpha)
    p = torch.Tensor(beta)
    def torch_squared_distances(x, y):
        return torch.sum((x[:,None,:] - y[None,:,:]) ** 2, 2)

    def torch_differences(x, y):
        return (x.t()[:,:,None] - y.t()[:,None,:])

    # A=exp(-|x_i - y_j|^2/(ker^2)).
    A = torch.exp(-torch_squared_distances(x, y) / (sigma ** 2))

    # B=2*(x_i - y_j)*exp(-|x_i - y_j|^2/(ker^2))/(ker^2).
    B = (torch_differences(x, y) * A)

    gamma_torch = -(torch.sum( pp * (torch.matmul(B,p)),2) / (0.5 * sigma ** 2)).t()
    """
    # compare output
    print("\nCuda gradconvolution :")
    print(gamma)

    print("\nNumpy gradconvolution :")
    print(gamma_py)

    print("\nIs everything okay ? ")
    print(np.allclose(gamma, gamma_py, atol = 1e-6))
