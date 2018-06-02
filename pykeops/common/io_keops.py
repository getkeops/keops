import numpy as np

import os.path
import importlib

from hashlib import sha256

from pykeops.common.get_options import get_backend, default_cuda_type
from pykeops.common.compile_routines import compile_generic_routine2
from pykeops import build_folder, script_folder, dll_prefix, dll_ext

# GENERIC FORMULAS DLLs =======================================================   



def get_generic_reduction(aliases, formula, cuda_type, sum_index, backend):

    dll_name = create_name(formula,aliases,cuda_type)

    compile_generic_routine2(aliases, formula, dll_name, cuda_type)

    importlib.import_module(dll_name)

    return dll_name.gen_red



def generic_reduction(formula, signature, result, *args,
                      backend="auto",
                      aliases=[], sum_index=0,
                      cuda_type=default_cuda_type):

    # Infer if we're working with numpy arrays or torch tensors from result's type :
    if hasattr(result, "ctypes"):  # Assume we're working with numpy arrays
        from pykeops.numpy.utils import ndims, to_ctype_pointer, is_on_device, dtype
        
    elif hasattr(result, "data_ptr"):  # Assume we're working with torch tensors
        from pykeops.torch.utils import ndims, to_ctype_pointer, is_on_device, dtype

    else:
        raise TypeError("result should either be a numpy array or a torch tensor.")

    backend = get_backend(backend,result,variables) 

    # Let's use our GPU, which works "in place" : ---------------------------------
    routine = get_generic_reduction(aliases, formula, cuda_type, sum_index, backend)
    result = routine(nx, ny, 1, 1, args)






















# Compose the DLL name ----------------------------------------------------
def create_name(formula, alisases,cuda_type, sum_index, backend):
    formula = formula.replace(" ", "")  # Remove spaces
    aliases = [alias.replace(" ", "") for alias in aliases]

    # Since the OS prevents us from using arbitrary long file names, an okayish solution is to call
    # a standard hash function, and hope that we won't fall into a non-injective nightmare case...
    dll_name = ",".join(aliases + [formula]) + "_" + cuda_type
    dll_name = sha256(dll_name.encode("utf-8")).hexdigest()
    return dll_name

