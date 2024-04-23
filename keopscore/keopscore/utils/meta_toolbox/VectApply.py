from .c_instruction import c_instruction
from .misc import Meta_Toolbox_Error
from .c_for import c_for_loop
from .c_variable import c_variable
from .c_array import c_array
from .c_lvalue import c_lvalue
from .c_expression import c_expression


def VectApply(fun, *args):
    # returns C++ code string to apply a scalar operation to fixed-size arrays, following broadcasting rules.
    # - fun is the scalar unary function to be applied, it must accept two c_variable or c_array inputs and output a string
    # - out must be a c_array instance
    # - args may be c_array or c_variable instances
    #
    # Example : if out.dim = 3, arg0.dim = 1, arg1.dim = 3,
    # it will generate the following (in pseudo-code for clarity) :
    #   #pragma unroll
    #   for(signed long int k=0; k<out.dim; k++)
    #       fun(out[k], arg0[0], arg1[k]);
    #
    # Equivalently, if out.dim = 3, arg0 is c_variable, arg1.dim = 3,
    # it will generate the following (in pseudo-code for clarity) :
    #   #pragma unroll
    #   for(signed long int k=0; k<out.dim; k++)
    #       fun(out[k], arg0, arg1[k]);
    
    dims = []
    for arg in args:
        if isinstance(arg, c_expression):
            dims.append(1)
        elif isinstance(arg, c_array):
            dims.append(arg.dim)
        else:
            Meta_Toolbox_Error("args must be c_expression, or c_array instances")
    dimloop = max(dims)
    if not set(dims) in ({dimloop}, {1, dimloop}):
        Meta_Toolbox_Error("incompatible dimensions in VectApply")
    incr_args = list((1 if dim == dimloop else 0) for dim in dims)

    forloop, k = c_for_loop(0, dimloop, 1, pragma_unroll=True)

    argsk = []
    for arg, incr in zip(args, incr_args):
        if isinstance(arg, c_expression):
            argsk.append(arg)
        elif isinstance(arg, c_array):
            argsk.append(arg[k * incr])
    body = fun(*argsk)
    if isinstance(body, str):
        body = c_instruction(body, set(), set())
    return forloop(body)
