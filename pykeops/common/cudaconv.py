import numpy as np

import ctypes
from ctypes import POINTER, c_float, c_int, cast

import os.path

from hashlib import sha256

from pykeops.common.get_options import get_backend, default_cuda_type
from pykeops.common.compile_routines import compile_generic_routine
from pykeops import build_folder, script_folder, dll_prefix, dll_ext

# GENERIC FORMULAS DLLs =========================================================================

__cuda_convs_generic = {}


def get_cuda_conv_generic(aliases, formula, cuda_type, sum_index, backend):
    """
    Returns the appropriate CUDA routine, given:
    - a list of aliases (strings)
    - a formula         (string)
    - a cuda_type       ("float" or "double")
    - a sum index       ( 0 for a sum over j, result indexed by i,
                          1 for a sum over i, result indexed by j)
    - a backend         (one of "CPU", "GPU_1D_host",   "GPU_2D_host",
                                       "GPU_2D_device", "GPU_2D_device" )

    If it is not already in __cuda_convs_generic, load it from the appropriate "build" folder.
    If the .dll/.so cannot be found, compile it on-the-fly (and store it for later use).
    """

    # Compose the DLL name ----------------------------------------------------------------------
    formula = formula.replace(" ", "")  # Remove spaces
    aliases = [alias.replace(" ", "") for alias in aliases]

    # Since the OS prevents us from using arbitrary long file names, an okayish solution is to call
    # a standard hash function, and hope that we won't fall into a non-injective nightmare case...
    dll_name = ",".join(aliases + [formula]) + "_" + cuda_type
    dll_name = sha256(dll_name.encode("utf-8")).hexdigest()

    if dll_name in __cuda_convs_generic:  # If this formula has already been loaded in memory...
        return __cuda_convs_generic[dll_name][backend][sum_index]
    else:  # Otherwise :
        # Load the DLL --------------------------------------------------------------------------

        dllabspath = build_folder + dll_name + dll_ext

        try:
            dll = ctypes.CDLL(dllabspath , mode=ctypes.RTLD_GLOBAL)
        except OSError:
            compile_generic_routine(aliases, formula, dll_name, cuda_type)
            dll = ctypes.CDLL(dllabspath, mode=ctypes.RTLD_GLOBAL)
            print("Loaded.")

        # These are all the C++ routines defined in "link_autodiff.cu" :
        routine_CPU_i = dll.CpuConv
        routine_CPU_j = dll.CpuTransConv

        routine_CPU_i.argtypes = [c_int, c_int, POINTER(c_float), POINTER(POINTER(c_float))]
        routine_CPU_j.argtypes = [c_int, c_int, POINTER(c_float), POINTER(POINTER(c_float))]

        # Add our new functions to the module's dictionnary :
        __cuda_convs_generic[dll_name] = {"CPU": [routine_CPU_i, routine_CPU_j]}

        # Avoid error if the lib was not compiled with cuda
        try: 
            # These are all the CUDA routines defined in "link_autodiff.cu" :
            routine_GPU_host_1D_i = dll.GpuConv1D
            routine_GPU_host_1D_j = dll.GpuTransConv1D
            routine_GPU_host_2D_i = dll.GpuConv2D
            routine_GPU_host_2D_j = dll.GpuTransConv2D
            routine_GPU_device_1D_i = dll.GpuConv1D_FromDevice
            routine_GPU_device_1D_j = dll.GpuTransConv1D_FromDevice
            routine_GPU_device_2D_i = dll.GpuConv2D_FromDevice
            routine_GPU_device_2D_j = dll.GpuTransConv2D_FromDevice
            
            routine_GPU_host_1D_i.argtypes = [c_int, c_int, POINTER(c_float), POINTER(POINTER(c_float))]
            routine_GPU_host_1D_j.argtypes = [c_int, c_int, POINTER(c_float), POINTER(POINTER(c_float))]
            routine_GPU_host_2D_i.argtypes = [c_int, c_int, POINTER(c_float), POINTER(POINTER(c_float))]
            routine_GPU_host_2D_j.argtypes = [c_int, c_int, POINTER(c_float), POINTER(POINTER(c_float))]
            routine_GPU_device_1D_i.argtypes = [c_int, c_int, POINTER(c_float), POINTER(POINTER(c_float))]
            routine_GPU_device_1D_j.argtypes = [c_int, c_int, POINTER(c_float), POINTER(POINTER(c_float))]
            routine_GPU_device_2D_i.argtypes = [c_int, c_int, POINTER(c_float), POINTER(POINTER(c_float))]
            routine_GPU_device_2D_j.argtypes = [c_int, c_int, POINTER(c_float), POINTER(POINTER(c_float))]

            __cuda_convs_generic[dll_name].update({
                 "GPU_1D_host": [routine_GPU_host_1D_i, routine_GPU_host_1D_j],
                 "GPU_2D_host": [routine_GPU_host_2D_i, routine_GPU_host_2D_j],
                 "GPU_1D_device": [routine_GPU_device_1D_i, routine_GPU_device_1D_j],
                 "GPU_2D_device": [routine_GPU_device_2D_i, routine_GPU_device_2D_j] })
        except AttributeError:
            # we do not have the Cuda routines, this is ok only if the backend is "CPU"
            if backend not in ["auto","CPU"]:
                raise ValueError('Cuda routines are not available.')
            
        return __cuda_convs_generic[dll_name][backend][sum_index]  # And return it.


# Ideally, this routine could be implemented by Joan :
def cuda_conv_generic(formula, signature, result, *args,
                      backend="auto",
                      aliases=[], sum_index=0,
                      cuda_type=default_cuda_type):
    """
    Executes the "autodiff" kernel associated to "formula".
    Backend is one of "auto", "CPU", "GPU", "GPU_1D" or "GPU_2D",
        and will be reassigned to "CPU", "GPU_1D_host",   "GPU_2D_host",
        "GPU_1D_device", "GPU_2D_device", depending on input data (see get_options.py)

    Aliases can be given as a list of strings.
    sum_index specifies whether the summation should be done over "I/X" (sum_index=1) or "J/Y" (sum_index=0).
    The arguments are given as :
        variables, sorted in the order specified by the "Var<index,dimension,I-or-J-or-P>" syntax.
    For instance,
        ```
        aliases = [ "DIMPOINT = 3", "DIMVECT = 4", "DIMOUT = 5",
                    "X = Var<1,DIMPOINT,0>" ,
                    "Y = Var<2,DIMPOINT,1>" ,
                    "U = Var<3,DIMVECT ,0>" ,
                    "V = Var<4,DIMVECT ,1>" ,
                    "B = Var<5,DIMOUT  ,1>" ,
                    "C = Param<0,1>"          ]
        formula = "Scal< Square<Scalprod<U,V>>, " \
                + "Scal< Exp< Scal<C, Minus<SqNorm2<Subtract<X,Y>>> > >,  B> >"
        cuda_conv_generic( formula, signature,
                           R, C, X, Y, U, V, B,
                           aliases = aliases )
        ```
    is a legal call, where :
    - R is a nx-by-5 float array (the output array)
    - C is a scalar
    - X is a nx-by-3 float array
    - Y is a ny-by-3 float array
    - U is a nx-by-4 float array
    - V is a ny-by-4 float array
    - B is a ny-by-5 float array

    (nx and ny are automatically inferred from the data;
    an error is thrown if the lengths of the input arrays are not compatible with each other)

    If the CUDA kernel associated to the given formula is not found in the "build/" folder,
    the routine is compiled on-the-fly using the "compile" script.

    N.B.: additional examples documenting the use of symbolic differentiation :

    Gradient with respect to X : ---------------------------------------------------------------
        ```
        aliases_gx = aliases + [ "Eta = Var<5,DIMOUT,0>" ]
        formula_gx = "Grad< " + formula + ", X, Eta>"
        cuda_conv_generic( formula,
                           R, C, X, Y, U, V, B, E,
                           aliases = aliases, sum_index = 0 )
        ```
    where :
    - R is a nx-by-3 float array (same as X)
    - E is a nx-by-5 float array (same as the output of "formula")


    Gradient with respect to V : ---------------------------------------------------------------
        ```
        aliases_gv = aliases + [ "Eta = Var<5,DIMOUT,0>" ]
        formula_gv = "Grad< " + formula + ", V, Eta>"
        cuda_conv_generic( formula,
                           R, C, X, Y, U, V, B, E,
                           aliases = aliases, sum_index = 1 )
        ```
    where :
    - R is a ny-by-4 float array (same as V)
    - E is a nx-by-5 float array (same as the output of "formula")

    """
    # Infer if we're working with numpy arrays or torch tensors from result's type :
    if hasattr(result, "ctypes"):  # Assume we're working with numpy arrays
        from pykeops.numpy.utils import ndims, to_ctype_pointer, is_on_device, dtype
        
    elif hasattr(result, "data_ptr"):  # Assume we're working with torch tensors
        from pykeops.torch.utils import ndims, to_ctype_pointer, is_on_device, dtype
    else:
        raise TypeError("result should either be a numpy array or a torch tensor.")

    # Check that *args matches the given signature ----------------------------------------------
    variables = []
    # Float32 ? Float64 ? The inputs' dtypes should be compatible with each other, and the output:
    out_dtype = dtype(result)
    nx = -1 # Length of the "i" variables
    ny = -1 # Length of the "j" variables
    for (var_id, (arg, sig)) in enumerate(zip(args, signature[1:])):  # Signature = [ Result, *Args]
        if not (dtype(arg) == out_dtype) : raise TypeError(
            "The dtype of the {:d}th input does not match that of the output tensor: {:s} vs {:s}.".format(
                var_id, str(dtype(arg)), str(out_dtype)) )

        if sig[1] == 0:  # If the current arg is an "X^n_i" variable
            if not (ndims(arg) == 2):          raise ValueError("Generic routines require 2D-arrays as variables.")
            if nx == -1: nx = arg.shape[0]  # First "X^0_i" variable encountered
            if not (nx == arg.shape[0]): raise ValueError(
                "CAT=0 variables (X_i) lengths are not compatible with each other.")
            if not (sig[0] == arg.shape[1]): raise ValueError(
                "The size of a CAT=0 variable does not match the signature.")
            variables.append(arg)  # No worries : arg is in fact a pointer, so no copy is done here

        elif sig[1] == 1:  # If the current arg is an "Y^m_j" variable
            if not (ndims(arg) == 2):          raise ValueError("Generic routines require 2D-arrays as variables.")
            if ny == -1: ny = arg.shape[0]  # First "Y^0_j" variable encountered
            if not (ny == arg.shape[0]): raise ValueError(
                "CAT=1 variables (Y_j) lengths are not compatible with each other.")
            if not (sig[0] == arg.shape[1]): raise ValueError(
                "The size of a CAT=1 variable does not match the signature.")
            variables.append(arg)  # No worries : arg is in fact a pointer, so no copy is done here

        elif sig[1] == 2:  # If the current arg is a parameter
            if not (sig[0] == arg.shape[0]): raise ValueError(
                "The size of a CAT=2 variable does not match the signature.")
            variables.append(arg)

    # Assert that we won't make an "empty" convolution :
    if not nx > 0: raise ValueError("There should be at least one (nonempty...) 'X_i' variable as input.")
    if not ny > 0: raise ValueError("There should be at least one (nonempty...) 'Y_j' variable as input.")

    # Check the result's shape :
    sig = signature[0]  # Signature = [ Result, *Args]
    if sig[1] == 2: raise ValueError("Derivatives wrt. parameters have not been implemented yet.")
    if not ndims(result) == 2: raise ValueError("The result array should be bi-dimensional.")
    if not sig[0] == result.shape[1]: raise ValueError("The width of the result array does not match the signature.")

    if sum_index == 0:  # Sum wrt. j, final result index by i
        if not sig[1] == 0: raise ValueError("The result's signature does not indicate an indexation by 'i'...")
        if not nx == result.shape[0]: raise ValueError(
            "The result array does not have the correct number of lines wrt. the 'X_i' inputs given.")

    if sum_index == 1:  # Sum wrt. i, final result index by j
        if not sig[1] == 1: raise ValueError("The result's signature does not indicate an indexation by 'j'...")
        if not ny == result.shape[0]: raise ValueError(
            "The result array does not have the correct number of lines wrt. the 'Y_j' inputs given.")

    # From python to C float pointers and int : -------------------------------------------------
    vars_p = tuple(to_ctype_pointer(var) for var in variables)
    vars_p = (POINTER(c_float) * len(vars_p))(*vars_p)

    result_p = to_ctype_pointer(result)
    
    backend = get_backend(backend,result,variables) 

    # Let's use our GPU, which works "in place" : -----------------------------------------------
    # N.B.: depending on sum_index, we're going to load "GpuConv" or "GpuTransConv",
    #       which make a summation wrt. 'j' or 'i', indexing the final result with 'i' or 'j'.
    routine = get_cuda_conv_generic(aliases, formula, cuda_type, sum_index, backend)
    routine(nx, ny, result_p, vars_p)
