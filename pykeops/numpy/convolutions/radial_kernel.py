from pykeops.numpy.specific_conv import load_keops
from pykeops import default_cuda_type


def radial_kernel_conv(x, y, beta, sigma, kernel = "gaussian", cuda_type=default_cuda_type):
    """
    Implements the operation :

    (x_i, y_j, beta_j)  ->  ( \sum_j k(x_i,y_j) beta_j )_i ,

    where k is a kernel function of parameter "sigma".
    Unlike a naive implementation, this code won't store in memory the matrix
    k(x_i,y_j) : it is therefore possible to use it when len(x) and len(y) are both large
    without getting a "memory overflow".

    N.B.: in an LDDMM setting, one would typically use "x = y = q", "beta = p". 
    """
    myconv = load_keops("radial_kernel_conv", cuda_type=cuda_type)
    return myconv.specific_conv(x, y, beta, sigma, kernel)


def radial_kernel_grad1conv(a, x, y, beta, sigma, kernel = "gaussian", cuda_type=default_cuda_type):
    """
    Implements the operation :

    (x_i, y_j, beta_j)  ->  ( \sum_j \partial_x k(x_i,y_j) beta_j )_i ,

    where k is a kernel function of parameter "sigma".
    Unlike a naive implementation, this code won't store in memory the matrix
    k(x_i,y_j) : it is therefore possible to use it when len(x) and len(y) are both large
    without getting a "memory overflow".

    N.B.: in an LDDMM setting, one would typically use "x = y = q", "beta = p".
    """
    myconv = load_keops("radial_kernel_grad1conv", cuda_type=cuda_type)
    return myconv.specific_grad1conv(a, x, y, beta, sigma, kernel)