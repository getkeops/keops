import numpy as np
import ctypes
from ctypes import *

#nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=192"  -Xcompiler -fPIC -shared -o cuda_conv.so cuda_conv.cu

# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_gradconv():
    dll = ctypes.CDLL('/home/bcharlier/abc/libds/cuda_gradconv.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.GaussGpuGrad1Conv
    func.argtypes = [c_float, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int, c_int]
    return func

# create __cuda_gradconv function with get_cuda_gradconv()
__cuda_gradconv = get_cuda_gradconv()

# convenient python wrapper for __cuda_gradconv it does all job with types convertation from python ones to C++ ones 
def cuda_gradconv(alpha,x, y, beta, result, sigma):
    alpha_p = alpha.ctypes.data_as(POINTER(c_float))
    x_p = x.ctypes.data_as(POINTER(c_float))
    y_p = y.ctypes.data_as(POINTER(c_float))
    beta_p = beta.ctypes.data_as(POINTER(c_float))
    result_p = result.ctypes.data_as(POINTER(c_float))

    ooSigma2 = float(1/ (sigma*sigma))
    
    nx = x.shape[0]
    ny = y.shape[0]

    dimPoint = x.shape[1]
    dimVect = beta.shape[1]

    __cuda_gradconv(ooSigma2, alpha_p, x_p, y_p, beta_p, result_p, dimPoint, dimVect, nx, ny )

# testing, benchmark gradconvolution with 
if __name__ == '__main__':

    np.set_printoptions(linewidth=200)
    
    sizeX=int(6)
    sizeY=int(10)
    dimPoint=int(3)
    dimVect=int(3)
    sigma = float(2)

    alpha = np.random.rand(sizeX,dimVect).astype('float32')
    x = np.random.rand(sizeX,dimPoint).astype('float32')
    y = np.random.rand(sizeY,dimPoint).astype('float32')
    beta = np.random.rand(sizeY,dimVect).astype('float32')

    # Call cuda kernel
    gamma = np.zeros(dimVect*sizeX).astype('float32')
    cuda_gradconv(alpha,x, y, beta, gamma, sigma)
    

    # A first implementation
    oosigma2= 1 / (sigma * sigma) 
    gamma_py = np.zeros((sizeX,dimVect)).astype('float32')

    for i in range(sizeX):
        for j in range(sizeY):
            rijk = (x[i,] - y[j,])
            rij2 = (rijk **2).sum()
            sga = (beta[j,] * alpha[i,]).sum()
            gamma_py[i,] -=  rijk * np.exp(-rij2 * oosigma2) * sga * 2.0 * oosigma2

    # compare output
    print("\nCuda convolution :")
    print(gamma.reshape((sizeX,dimVect)))
    
    print("\nPython gradconvolution 1 :")
    print(gamma_py)
