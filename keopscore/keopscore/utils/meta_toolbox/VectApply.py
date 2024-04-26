from .c_instruction import c_instruction, c_instruction_from_string
from .misc import Meta_Toolbox_Error
from .c_for import c_for_loop
from .c_variable import c_variable
from .c_array import c_array, c_fixed_size_array
from .c_lvalue import c_lvalue
from .c_expression import c_expression


def VectApply(fun, *args):
    # returns C++ code string to apply a scalar operation to c_arrays, following broadcasting rules.
    # - fun is the scalar unary function to be applied, it must accept c_expression inputs and outputs a c_instruction
    # - args must be c_array instances
    if not all(isinstance(arg, c_array) for arg in args):
        Meta_Toolbox_Error("inputs should be c_array instances")
    dims = [arg.dim for arg in args]
    dimloop = max(dims)
    if not set(dims) in ({dimloop}, {1, dimloop}):
        Meta_Toolbox_Error("incompatible dimensions in VectApply")
    forloop, k = c_for_loop(0, dimloop, 1, pragma_unroll=True)
    body = fun(*(arg[k] for arg in args))
    if isinstance(body, str):
        body = c_instruction_from_string(body)
    return forloop(body)
