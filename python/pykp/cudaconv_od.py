import numpy as np
import ctypes
from ctypes import *
import os.path

#nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=192"  -Xcompiler -fPIC -shared -o cuda_conv.so cuda_conv.cu

# extract cuda_conv function pointer in the shared object cuda_conv.so
def get_cuda_convs_od():
    """
    Loads the convolution routine from the compiled .so file.
    """
    dll_name = 'cuda_conv.so'
    dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep+ 'build' + os.path.sep + dll_name
    dll = ctypes.CDLL(dllabspath, mode=ctypes.RTLD_GLOBAL)

    func = dll.GaussGpuEvalConv_onDevice


    #(float ooSigma2, float* x_d, float* y_d, float* beta_d, float* gamma_d, int dimPoint, int dimVect, int nx, int ny)	
    func.argtypes = [c_float, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int,   c_int,     c_int, c_int]
    return func

# create __cuda_conv function with get_cuda_conv()
__cuda_convs_od = get_cuda_convs_od()

def cuda_conv_od(x, y, beta, result, sigma, kernel = "gaussian"):
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
    x_p      = ctypes.byref( ctypes.c_float.from_address(x.data_ptr()))
    y_p      = ctypes.byref( ctypes.c_float.from_address( y.data_ptr()))
    beta_p   = ctypes.byref( ctypes.c_float.from_address(beta.data_ptr()))
    result_p = ctypes.byref( ctypes.c_float.from_address(result.data_ptr()))

    nx = x.shape[0] ; ny = y.shape[0]

    dimPoint =    x.shape[1]
    dimVect  = beta.shape[1]

    ooSigma2 = float(1/ (sigma*sigma)) # Compute this once and for all

    # Let's use our GPU, which works "in place" :
    __cuda_convs_od(ooSigma2, x_p, y_p, beta_p, result_p, dimPoint, dimVect, nx, ny )

if __name__ == '__main__':
    """
        testing, benchmark convolution with two naive python implementations of the Gaussian convolution
        """
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

k='gaussian'

g0 = torch.mm(torch_kernel(xc,yc,sc,kernel=k),bc).cpu().numpy()

g1 = torch.from_numpy(np.zeros([N*E]).astype('float32')).cuda()
cuda_conv_od(xc, yc, bc, g1, s, kernel=k)

print(g0)
print(g1.cpu().view(N,E))
