import numpy as np
import ctypes
from ctypes import *
import os.path

#nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=192"  -Xcompiler -fPIC -shared -o cuda_conv.so cuda_conv.cu

# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_conv():
    dll_name = "cuda_conv.so"
    dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../src/cuda/' + dll_name
    dll = ctypes.CDLL(dllabspath, mode=ctypes.RTLD_GLOBAL)
    func = dll.GaussGpuEvalConv
    func.argtypes = [c_float, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int, c_int]
    return func

# create __cuda_conv function with get_cuda_conv()
__cuda_conv = get_cuda_conv()

# convenient python wrapper for __cuda_conv it does all job with types convertation from python ones to C++ ones 
def cuda_conv(x, y, beta, result, sigma):
    x_p = x.ctypes.data_as(POINTER(c_float))
    y_p = y.ctypes.data_as(POINTER(c_float))
    beta_p = beta.ctypes.data_as(POINTER(c_float))
    result_p = result.ctypes.data_as(POINTER(c_float))

    ooSigma2 = float(1/ (sigma*sigma))
    
    nx = x.shape[0]
    ny = y.shape[0]

    dimPoint = x.shape[1]
    dimVect = beta.shape[1]

    __cuda_conv(ooSigma2, x_p, y_p, beta_p, result_p, dimPoint, dimVect, nx, ny )

# testing, benchmark convolution with 
if __name__ == '__main__':

    np.set_printoptions(linewidth=200)
    
    sizeX=int(6)
    sizeY=int(10)
    dimPoint=int(3)
    dimVect=int(3)
    sigma = float(2)

    x = np.random.rand(sizeX,dimPoint).astype('float32')
    y = np.random.rand(sizeY,dimPoint).astype('float32')
    beta = np.random.rand(sizeY,dimVect).astype('float32')
    # x = np.sin([ np.arange(float(sizeX)), np.arange(float(sizeX))* 2,np.arange(float(sizeX))**2] ).astype('float32')
    # y = np.cos([ np.arange(float(sizeY)), np.arange(float(sizeY))* 2,np.arange(float(sizeY))**2]  ).astype('float32')
    # beta =np.array(np.arange(float(dimVect*sizeY)) +1).reshape(dimVect,sizeY).astype('float32')

    # Call cuda kernel
    gamma = np.zeros(dimVect*sizeX).astype('float32')
    cuda_conv(x, y, beta, gamma, sigma)
    

    # A first implementation
    oosigma2= 1 / (sigma * sigma) 
    gamma_py = np.zeros((sizeX,dimVect)).astype('float32')

    for i in range(sizeX):
        for j in range(sizeY):
            rij2 = 0.
            for k in range(dimPoint):
                rij2 += (x[i,k] - y[j,k]) ** 2
            for l in range(dimVect):
                gamma_py[i,l] +=  np.exp(-rij2 * oosigma2) * beta[j,l]

    # A second implementation
    r2 = np.zeros((sizeX,sizeY)).astype('float32')
    for i in range(sizeX):
        for j in range(sizeY):
            for k in range(dimPoint):
                r2[i,j] += (x[i,k] - y[j,k]) ** 2
    
    K = np.exp(-r2 * oosigma2)
    gamma_py2 = np.dot(K,beta)

    # compare output
    print("\nCuda convolution :")
    print(gamma.reshape((sizeX,dimVect)))
    
    print("\nPython convolution 1 :")
    print(gamma_py)
    
    print("\nPython convolution 2 :")
    print(gamma_py2)



