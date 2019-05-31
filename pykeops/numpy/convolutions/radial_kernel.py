import importlib.util
import os

from pykeops import bin_folder, build_type
from pykeops.common.compile_routines import compile_specific_conv_routine
from pykeops.common.utils import create_and_lock_build_folder
from pykeops.numpy import default_dtype


class LoadKeopsSpecific:
    r"""
    This class compile the cuda routines if necessary and load it via the method import_module()
    """
    
    def __init__(self, dllname, dtype=default_dtype):
        self.dll_name = dllname
        self.dtype = dtype
        
        spec = importlib.util.find_spec(dllname)
        
        if (spec is None) or (build_type == 'Debug'):
            self.build_folder = bin_folder + os.path.sep + 'build-' + self.dll_name
            self._safe_compile()
    
    @create_and_lock_build_folder()
    def _safe_compile(self):
        compile_specific_conv_routine(self.dll_name, self.dtype, build_folder=self.build_folder)
    
    def import_module(self):
        return importlib.import_module(self.dll_name)


class RadialKernelConv:
    r"""
    Implements the operation :

    (x_i, y_j, beta_j)  ->  ( \sum_j k(x_i,y_j) beta_j )_i ,

    where k is a kernel function of parameter "sigma".
    Unlike a naive implementation, this code won't store in memory the matrix
    k(x_i,y_j) : it is therefore possible to use it when len(x) and len(y) are both large
    without getting a "memory overflow".

    N.B.: in an LDDMM setting, one would typically use "x = y = q", "beta = p". 
    """
    
    def __init__(self, dtype=default_dtype, cuda_type=None):
        if cuda_type:
            # cuda_type is just old keyword for dtype, so this is just a trick to keep backward compatibility
            dtype = cuda_type
        self.myconv = LoadKeopsSpecific('radial_kernel_conv', dtype).import_module()
    
    def __call__(self, x, y, beta, sigma, kernel='gaussian'):
        return self.myconv.specific_conv(x, y, beta, sigma, kernel)


class RadialKernelGrad1conv:
    r"""
    Implements the operation :

    (x_i, y_j, beta_j)  ->  ( \sum_j \partial_x k(x_i,y_j) beta_j )_i ,

    where k is a kernel function of parameter "sigma".
    Unlike a naive implementation, this code won't store in memory the matrix
    k(x_i,y_j) : it is therefore possible to use it when len(x) and len(y) are both large
    without getting a "memory overflow".

    N.B.: in an LDDMM setting, one would typically use "x = y = q", "beta = p".
    """
    
    def __init__(self, dtype=default_dtype, cuda_type=None):
        if cuda_type:
            # cuda_type is just old keyword for dtype, so this is just a trick to keep backward compatibility
            dtype = cuda_type
        if cuda_type:
            # cuda_type is just old keyword for dtype, so this is just a trick to keep backward compatibility
            dtype = cuda_type
        self.myconv = LoadKeopsSpecific('radial_kernel_grad1conv', dtype).import_module()
    
    def __call__(self, a, x, y, beta, sigma, kernel='gaussian'):
        return self.myconv.specific_grad1conv(a, x, y, beta, sigma, kernel)
