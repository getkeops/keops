import importlib

from pykeops import build_type, default_cuda_type
from pykeops.common.compile_routines import compile_generic_routine2
from pykeops.common.get_options import get_tag_backend
from pykeops.common.utils import create_name, axis2cat


def load_keops(formula, aliases, cuda_type):
    # create the name from formula, alisas and cuda_type.
    dll_name = create_name(formula, aliases, cuda_type)
    
    # Import and compile
    compile = (build_type == 'Debug')
    
    if not compile:
        try:
            myconv = importlib.import_module(dll_name)
        except ImportError:
            compile = True
    
    if compile:
        compile_generic_routine2(formula, aliases, dll_name, cuda_type)
        myconv = importlib.import_module(dll_name)
        print("Loaded.")
    return myconv


def genred(formula, aliases, *args, axis = 0, backend="auto", cuda_type=default_cuda_type):
    myconv = load_keops(formula, aliases, cuda_type)

    # Get tags
    tagIJ = (axis + 1)%2  # tagIJ=0 means sum over j, tagIJ=1 means sum over j
    tagCpuGpu, tag1D2D, _ = get_tag_backend(backend, args)

    # Perform computation using KeOps
    result = myconv.gen_red(tagIJ, tagCpuGpu, tag1D2D, *args)
    return result


def genred_pytorch(formula, aliases, *args, axis=0, backend="auto", cuda_type=default_cuda_type):
    myconv = load_keops(formula, aliases, cuda_type)

    tagIJ = axis2cat(axis)  # tagIJ=0 means sum over j, tagIJ=1 means sum over j
    _, tag1D2D, tagHostDevice = get_tag_backend(backend, args)

    # Perform computation using KeOps
    print('\n======   DEBUG   ==========')
    print('Compiled formula :', myconv.formula)
    print('Called formula : ', formula)
    print('Nbr of args in : ', myconv.nargs)
    print('Dim of Output : ', myconv.dimout)
    print('\n=======   DEBUG   =========')
    result = myconv.gen_red_from_device(tagIJ, tag1D2D, tagHostDevice, *args)
    return result


def nargs(formula, aliases, cuda_type):
    myconv = load_keops(formula, aliases, cuda_type)
    return myconv.nargs


def dimout(formula, aliases, cuda_type):
    myconv = load_keops(formula, aliases, cuda_type)
    return myconv.dimout
