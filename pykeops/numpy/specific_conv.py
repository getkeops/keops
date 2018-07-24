import importlib

from pykeops import build_type, default_cuda_type
from pykeops.common.compile_routines import compile_specific_conv_routine


# extract radial_kernels_conv function pointer in the shared object radial_kernels_conv.so
def load_keops(target, cuda_type=default_cuda_type):
    dll_name = target
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
        print("Loaded.")
    return myconv