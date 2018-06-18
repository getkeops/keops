import importlib

from pykeops import build_type, default_cuda_type
from pykeops.common.compile_routines import compile_generic_routine2
from pykeops.common.get_options import get_backend
from pykeops.common.utils import create_name

def load_keops(formula, aliases, cuda_type = default_cuda_type):
    
    # create the name from formula, alisasa anf cuda_type.
    dll_name = create_name(formula,aliases,cuda_type)
    
    # Import and compile
    compile = (build_type == 'Debug')
    
    if (not compile):
        try:
            myconv = importlib.import_module(dll_name)
        except ImportError:
            compile = True
    
    if compile:
        compile_generic_routine2(aliases, formula, dll_name, cuda_type)
        myconv = importlib.import_module(dll_name)
        print("Loaded.")
    return myconv

def genred(formula, aliases, *args, sum_index = 0, backend = "auto", cuda_type = default_cuda_type):
    
    myconv = load_keops(formula,aliases,cuda_type)

    # Perform computation using KeOps
    tagIJ = 0;# tagIJ=0 means sum over j, tagIJ=1 means sum over j
    tagCpuGpu = 0;# tagCpuGpu=0 means convolution on Cpu, tagCpuGpu=1 means convolution on Gpu, tagCpuGpu=2 means convolution on Gpu from device data
    tag1D2D = 0;# tag1D2D=0 means 1D Gpu scheme, tag1D2D=1 means 2D Gpu scheme
    result = myconv.gen_red(tagIJ, tagCpuGpu, tag1D2D, *args)

    return result


def genred_fromdevice(formula, aliases, *args, sum_index = 0, backend = "auto", cuda_type = default_cuda_type):
    
    myconv = load_keops(formula,aliases,cuda_type)

    # Perform computation using KeOps
    tagIJ = 0;# tagIJ=0 means sum over j, tagIJ=1 means sum over j
    tagCpuGpu = 0;# tagCpuGpu=0 means convolution on Cpu, tagCpuGpu=1 means convolution on Gpu, tagCpuGpu=2 means convolution on Gpu from device data
    tag1D2D = 0;# tag1D2D=0 means 1D Gpu scheme, tag1D2D=1 means 2D Gpu scheme
    result = myconv.gen_red_from_device(tagIJ, tagCpuGpu, tag1D2D, *args)

    return result

