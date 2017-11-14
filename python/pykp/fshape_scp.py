import numpy as np
import ctypes
from ctypes import *
import os.path

# extract cuda_fshape_scp function pointer in the shared object cuda_fshape_scp_*.so
def get_cuda_fshape_scp():
    """
    Loads the routine from the compiled .so file.
    """
    func_dict = {}

    for name_geom in ["gaussian","cauchy"] :
        for name_sig in ["gaussian","cauchy"] :
            for name_var in ["gaussian_unoriented","binet","gaussian_oriented","linear"] :
                name = name_geom + name_sig + name_var
                dll_name = "cuda_fshape_scp_" + name + ".so"
                dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep+ 'build' + os.path.sep + dll_name
                dll = ctypes.CDLL(dllabspath, mode=ctypes.RTLD_GLOBAL)
                func = dll.cudafshape
                # Arguments :     1/sx^2,   1/sf^2, 1/st^2,     x,               y,                f,                  g,              alpha,            beta,             result,           dim-xy, dim-fg,  dim-beta, nx,   ny
                func.argtypes = [c_double, c_double, c_double, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int,   c_int,     c_int, c_int]
                func_dict[name] = func
    return func_dict

# create __cuda_fshape_scp function with get_cuda_fshape_scp()
__cuda_fshape_scp = get_cuda_fshape_scp()

# convenient python wrapper for __cuda_fshape_scp it does all job with types convertation from python ones to C++ ones 
def cuda_shape_scp(x, y, alpha, beta, result, sigma_geom, sigma_var = 1, kernel_geom = "gaussian", kernel_var = "binet"):
    """
    Implements the operation :

    (x_i, y_j, beta_j)  ->  ( \sum_j k(x_i,y_j) beta_j )_i ,

    where k is a kernel function of parameter "sigma".
    Unlike a naive pytorch implementation, this code won't store in memory the matrix
    k(x_i,y_j) : it is therefore possible to use it when len(x) and len(y) are both large
    without getting a "memory overflow".

    N.B.: in an LDDMM setting, one would typically use "x = y = q", "beta = p". 
    """
    # From python to C float pointers and int :
    x_p      =      x.ctypes.data_as(POINTER(c_float))
    y_p      =      y.ctypes.data_as(POINTER(c_float))
    alpha_p  =      alpha.ctypes.data_as(POINTER(c_float))
    beta_p   =      beta.ctypes.data_as(POINTER(c_float))
    result_p =      result.ctypes.data_as(POINTER(c_float))

    nx = x.shape[0] ; ny = y.shape[0]

    dimPoint =    x.shape[1]
    dimVect  = beta.shape[1]

    ooSigma_geom2 = float(1/ (sigma_geom*sigma_geom)) # Compute this once and for all
    ooSigma_var2 = float(1/ (sigma_var*sigma_var)) # Compute this once and for all

    # the functional part is disarmed
    kernel_sig="gaussian"
    sigma_sig = 1/np.finfo(np.float32).eps # one over machine epsilon
    ooSigma_sig2 = float(1/ (sigma_sig*sigma_sig)) # Compute this once and for all
    name = kernel_geom + kernel_sig + kernel_var
    dimSig = int(1);
    f_p = np.zeros([nx,1]).ctypes.data_as(POINTER(c_float))
    g_p = np.zeros([ny,1]).ctypes.data_as(POINTER(c_float))

    # Let's use our GPU, which works "in place" :
    __cuda_fshape_scp[name](ooSigma_geom2,ooSigma_sig2,ooSigma_var2, x_p, y_p, f_p, g_p, alpha_p, beta_p, result_p, dimPoint,dimSig, dimVect, nx, ny )



if __name__ == '__main__':
    """
    testing, benchmark convolution with two naive python implementations of the Gaussian convolution
    """
    np.set_printoptions(linewidth=200)

    sizeX    = int(4)
    sizeY    = int(5)
    dimPoint = int(3)
    dimVect  = int(3)
    sigma_geom    = float(2)
    sigma_var    = float(np.pi)

    if False : # Random test
        x    = np.random.rand(sizeX,dimPoint).astype('float32')
        y    = np.random.rand(sizeY,dimPoint).astype('float32')
        alpha = np.random.rand(sizeX,dimVect ).astype('float32')
        beta = np.random.rand(sizeY,dimVect ).astype('float32')
    else :    # Deterministic one
        x     =np.transpose(np.sin([ np.arange(float(sizeX)), np.arange(float(sizeX))* 2,np.arange(float(sizeX))**2] ).astype('float32'))
        y     =np.transpose(np.cos([ np.arange(float(sizeY)), np.arange(float(sizeY))* 2,np.arange(float(sizeY))**2] ).astype('float32'))
        alpha =1+np.transpose(np.tan([ np.arange(float(sizeX)) *4 , np.arange(float(sizeX)),np.arange(float(sizeX))**2] ).astype('float32'))
        beta  =1+np.transpose(np.tan([ np.arange(float(sizeY))*4, np.arange(float(sizeY))* 2,np.arange(float(sizeY))**2] ).astype('float32'))


    # Call cuda kernel
    gamma = np.zeros(sizeX).astype('float32')
    cuda_shape_scp(x, y,alpha, beta, gamma, sigma_geom, sigma_var) # In place, gamma_i = k(x_i,y_j) @ beta_j

    # A first implementation, with (shock horror !) a bunch of "for" loops
    gamma_py = np.zeros(sizeX).astype('float32')

    for i in range(sizeX):
        for j in range(sizeY):
            rij2 = np.sum((x[i,:] - y[j,:]) ** 2)
            sij2 = np.sum((alpha[i,:] *  beta[j,:]) )** 2 / np.sqrt(np.sum(alpha[i,:] **2) * np.sum(beta[j,:] **2)) 
            gamma_py[i] +=  np.exp(-rij2 / sigma_geom**2) * sij2


    # compare output
    print("\nCuda convolution :")
    print(gamma)

    print("\nPython convolution 1 :")
    print(gamma_py)

    print("\nIs everything okay ? ")
    print(np.allclose(gamma, gamma_py ))
