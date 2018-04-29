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
