import importlib

from pykeops import build_type, default_cuda_type
from pykeops.common.utils import c_type
from pykeops.common.compile_routines import compile_specific_fshape_scp_routine


# extract radial_kernels_conv function pointer in the shared object radial_kernels_conv.so
def load_keops(target,
               kernel_geom="gaussian", kernel_sig="gaussian", kernel_sphere="binet",
               cuda_type=default_cuda_type):

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