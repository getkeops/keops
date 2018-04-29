import numpy as np
import ctypes
from ctypes import POINTER, c_int, c_float

from pykeops import build_folder, script_folder, dll_prefix, dll_ext
from pykeops.common.compile_routines import compile_specific_routine

import os.path

# extract radial_kernels_conv function pointer in the shared object radial_kernels_conv.so
def get_specific_lib(target,signature):
    """
    Loads the convolution routine from the compiled .so file.
    """
    dllabspath = build_folder + os.path.sep + 'specific' + os.path.sep +  dll_prefix + target + dll_ext

    try:
        dll = ctypes.CDLL(dllabspath , mode=ctypes.RTLD_GLOBAL)
    except OSError:
        compile_specific_routine(dllname=target, cuda_type="float" )
        dll = ctypes.CDLL(dllabspath, mode=ctypes.RTLD_GLOBAL)
        print("Loaded.")

    func_dict = {}
    for (name, routine) in [("gaussian",  dll.GaussGpuEval), 
                            ("laplacian", dll.LaplaceGpuEval), 
                            ("cauchy",    dll.CauchyGpuEval), 
                            ("inverse_multiquadric",    dll.InverseMultiquadricGpuEval) ] :
        func = routine
        func.argtypes = signature
        func_dict[name] = func
    return func_dict

