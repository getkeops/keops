import numpy as np

from pykeops.numpy.specific_fshape_scp import load_keops
from pykeops import default_cuda_type


def fshape_scp(x, y, f, g, alpha, beta,
                       sigma_geom=1.0, sigma_sig=1.0, sigma_sphere=np.pi/2,
                       kernel_geom="gaussian", kernel_sig="gaussian", kernel_sphere="binet",
                       cuda_type=default_cuda_type):
    """
    Implements the operation :

    (x_i, y_j, beta_j)  ->  ( \sum_j k(x_i,y_j) beta_j )_i ,

    where k is a kernel function of parameter "sigma".
    Unlike a naive implementation, this code won't store in memory the matrix
    k(x_i,y_j) : it is therefore possible to use it when len(x) and len(y) are both large
    without getting a "memory overflow".

    N.B.: in an LDDMM setting, one would typically use "x = y = q", "beta = p".
    """
    myconv = load_keops("fshape_scp", kernel_geom, kernel_sig, kernel_sphere, cuda_type=cuda_type)
    return myconv.specific_fshape_scp(x, y, f, g, alpha, beta, sigma_geom , sigma_sig, sigma_sphere)
