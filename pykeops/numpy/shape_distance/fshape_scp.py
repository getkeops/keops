import numpy as np

import importlib

from pykeops import build_type, default_cuda_type
from pykeops.common.utils import c_type
from pykeops.common.compile_routines import compile_specific_fshape_scp_routine


class FshapeScp:
    """
    Implements the operation :

    (x_i, y_j, beta_j)  ->  ( \sum_j k(x_i,y_j) beta_j )_i ,

    where k is a kernel function of parameter "sigma".
    Unlike a naive implementation, this code won't store in memory the matrix
    k(x_i,y_j) : it is therefore possible to use it when len(x) and len(y) are both large
    without getting a "memory overflow".

    N.B.: in an LDDMM setting, one would typically use "x = y = q", "beta = p".
    """
    def __init__(self, kernel_geom="gaussian", kernel_sig="gaussian", kernel_sphere="binet", cuda_type=default_cuda_type):
        self.kernel_geom = kernel_geom
        self.kernel_sig = kernel_sig
        self.kernel_sphere = kernel_sphere
        self.cuda_type = cuda_type

    def __call__(self, x, y, f, g, alpha, beta, sigma_geom=1.0, sigma_sig=1.0, sigma_sphere=np.pi/2,):
        myconv = self.load_keops("fshape_scp", self.kernel_geom, self.kernel_sig, self.kernel_sphere, self.cuda_type)
        return myconv.specific_fshape_scp(x, y, f, g, alpha, beta, sigma_geom , sigma_sig, sigma_sphere)

    @staticmethod
    # extract radial_kernels_conv function pointer in the shared object radial_kernels_conv.so
    def load_keops(target, kernel_geom, kernel_sig, kernel_sphere, cuda_type):
        dllname = target + "_" + kernel_geom + kernel_sig + kernel_sphere + "_" + c_type[cuda_type]
        # Import and compile
        compile = (build_type == 'Debug')

        if not compile:
            try:
                myconv = importlib.import_module(dllname)
            except ImportError:
                compile = True

        if compile:
            compile_specific_fshape_scp_routine(dllname, kernel_geom, kernel_sig, kernel_sphere, cuda_type)
            myconv = importlib.import_module(dllname)
            print("Loaded.")
        return myconv