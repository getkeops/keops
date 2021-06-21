from keops.python_engine.utils.code_gen_utils import c_for_loop, new_c_varname, c_variable

def keops_abs(x):
    # returns the C++ code string for the abs function applied to a C++ variable
    # - x must be of type c_variable
    if x.dtype in ["float","double"]:
        return c_variable(x.dtype, f"abs({x.id})")
    else:
        raise ValueError("not implemented.")

def keops_cos(x):
    # returns the C++ code string for the cos function applied to a C++ variable
    # - x must be of type c_variable
    if x.dtype in ["float", "double"]:  # TODO: check CUDA_ARCH version
        return c_variable(x.dtype, f"cos({x.id})")
    else:
        raise ValueError("not implemented.")

def keops_sin(x):
    # returns the C++ code string for the sin function applied to a C++ variable
    # - x must be of type c_variable
    if x.dtype in ["float", "double"]:  # TODO: check CUDA_ARCH version
        return c_variable(x.dtype, f"sin({x.id})")
    else:
        raise ValueError("not implemented.")

def keops_acos(x):
    # returns the C++ code string for the acos function applied to a C++ variable
    # - x must be of type c_variable
    if x.dtype in ["float", "double"]:  # TODO: check CUDA_ARCH version
        return c_variable(x.dtype, f"acos({x.id})")
    else:
        raise ValueError("not implemented.")
        
def keops_atan2(y, x):
    # returns the C++ code string for the atan2 function applied to C++ variables
    # - x and y must be of type c_variable
    simple_dtypes = ["float", "double"]
    if (x.dtype in simple_dtypes) and (y.dtype in simple_dtypes):  # TODO: check CUDA_ARCH version
        return c_variable(x.dtype, f"atan2({y.id},{x.id})")
    else:
        raise ValueError("not implemented.")
                    
def keops_exp(x):
    # returns the C++ code string for the exponential function applied to a C++ variable
    # - x must be of type c_variable
    if x.dtype in ["float", "double"]:
        return c_variable(x.dtype, f"exp({x.id})")
    else:
        raise ValueError("not implemented.")
        
def keops_rcp(x):
    # returns the C++ code string for the inverse function applied to a C++ variable
    # - x must be of type c_variable
    if x.dtype in ["float", "double"]:
        return c_variable(x.dtype, f"(1.0f/({x.id}))")
    else:
        raise ValueError("not implemented.")
        
def keops_rsqrt(x):
    # returns the C++ code string for the inverse square root function applied to a C++ variable
    # - x must be of type c_variable
    if x.dtype in ["float", "double"]:  # TODO: check CUDA_ARCH version
        return c_variable(x.dtype, f"(1.0f/sqrt({x.id}))")
    else:
        raise ValueError("not implemented.")
                
def keops_sqrt(x):
    # returns the C++ code string for the square root function applied to a C++ variable
    # - x must be of type c_variable
    if x.dtype in ["float","double"]:
        return c_variable(x.dtype, f"sqrt({x.id})")
    else:
        raise ValueError("not implemented.")


