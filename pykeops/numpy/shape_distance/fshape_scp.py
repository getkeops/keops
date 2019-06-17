import importlib.util
import os

import numpy as np

from pykeops import bin_folder, build_type
from pykeops.common.compile_routines import compile_specific_fshape_scp_routine
from pykeops.common.utils import c_type, create_and_lock_build_folder
from pykeops.numpy import default_dtype


class LoadKeopsFshapeScp:
    r"""
    This class compile the cuda routines if necessary and load it via the method import_module()
    """
    
    def __init__(self, target, kernel_geom, kernel_sig, kernel_sphere, dtype):
        self.kernel_geom = kernel_geom
        self.kernel_sig = kernel_sig
        self.kernel_sphere = kernel_sphere
        self.dll_name = target + "_" + kernel_geom + kernel_sig + kernel_sphere + "_" + c_type[dtype]
        self.dtype = dtype
        
        spec = importlib.util.find_spec(self.dll_name)
        
        if (spec is None) or (build_type == 'Debug'):
            self.build_folder = bin_folder + os.path.sep + 'build-' + self.dll_name
            self._safe_compile()
    
    @create_and_lock_build_folder()
    def _safe_compile(self):
        compile_specific_fshape_scp_routine(self.dll_name, self.kernel_geom, self.kernel_sig, self.kernel_sphere,
                                            self.dtype, build_folder=self.build_folder)
    
    def import_module(self):
        return importlib.import_module(self.dll_name)


class FshapeScp:
    """
    Implements the operation :
    
    (x_i, y_j, beta_j)  ->  ( sum_j k(x_i,y_j) beta_j )_i ,
    
    where k is a kernel function of parameter "sigma".
    Unlike a naive implementation, this code won't store in memory the matrix
    k(x_i,y_j) : it is therefore possible to use it when len(x) and len(y) are both large
    without getting a "memory overflow".
    
    N.B.: in an LDDMM setting, one would typically use "x = y = q", "beta = p".
    """

    def __init__(self, kernel_geom="gaussian", kernel_sig="gaussian", kernel_sphere="binet", dtype=default_dtype,
                 cuda_type=None):
        if cuda_type:
            # cuda_type is just old keyword for dtype, so this is just a trick to keep backward compatibility
            dtype = cuda_type
        self.kernel_geom = kernel_geom
        self.kernel_sig = kernel_sig
        self.kernel_sphere = kernel_sphere
        self.dtype = dtype

    def __call__(self, x, y, f, g, alpha, beta, sigma_geom=1.0, sigma_sig=1.0, sigma_sphere=np.pi / 2, ):
        myconv = LoadKeopsFshapeScp("fshape_scp", self.kernel_geom, self.kernel_sig, self.kernel_sphere,
                                    self.dtype).import_module()
        return myconv.specific_fshape_scp(x, y, f, g, alpha, beta, sigma_geom, sigma_sig, sigma_sphere)
