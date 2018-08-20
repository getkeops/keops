import importlib

from pykeops import build_type, default_cuda_type
from pykeops.common.compile_routines import compile_specific_conv_routine


class RadialKernelConv:
    """
    Implements the operation :

    (x_i, y_j, beta_j)  ->  ( \sum_j k(x_i,y_j) beta_j )_i ,

    where k is a kernel function of parameter "sigma".
    Unlike a naive implementation, this code won't store in memory the matrix
    k(x_i,y_j) : it is therefore possible to use it when len(x) and len(y) are both large
    without getting a "memory overflow".

    N.B.: in an LDDMM setting, one would typically use "x = y = q", "beta = p". 
    """

    def __init__(self, cuda_type=default_cuda_type):
        self.myconv = load_keops('radial_kernel_conv', cuda_type)

    def __call__(self, x, y, beta, sigma, kernel='gaussian'):
        return self.myconv.specific_conv(x, y, beta, sigma, kernel)


class RadialKernelGrad1conv:
    """
    Implements the operation :

    (x_i, y_j, beta_j)  ->  ( \sum_j \partial_x k(x_i,y_j) beta_j )_i ,

    where k is a kernel function of parameter "sigma".
    Unlike a naive implementation, this code won't store in memory the matrix
    k(x_i,y_j) : it is therefore possible to use it when len(x) and len(y) are both large
    without getting a "memory overflow".

    N.B.: in an LDDMM setting, one would typically use "x = y = q", "beta = p".
    """
    def __init__(self, cuda_type=default_cuda_type):
        self.myconv = load_keops('radial_kernel_grad1conv', cuda_type)

    def __call__(self, a, x, y, beta, sigma, kernel='gaussian'):
        return self.myconv.specific_grad1conv(a, x, y, beta, sigma, kernel)


def load_keops(target, cuda_type=default_cuda_type):
    # Import and compile
    compile = (build_type == 'Debug')

    if not compile:
        try:
            myconv = importlib.import_module(target)
        except ImportError:
            compile = True

    if compile:
        compile_specific_conv_routine(target, cuda_type)
        myconv = importlib.import_module(target)
        print('Loaded.')
    return myconv