from .c_for import c_for_loop
from .c_array import c_array
from .c_lvalue import c_lvalue
from .misc import Meta_Toolbox_Error
from .c_expression import c_expression


def VectCopy(out, arg, dim=None):
    # returns a C++ code string representing a vector copy between fixed-size arrays
    # - dim is dimension of arrays
    # - out is c_lvalue or c_array representing the output array
    # - arg is c_expression or c_array representing the input array
    if not (isinstance(arg, c_expression) or isinstance(arg, c_array)):
        Meta_Toolbox_Error("arg must be c_expression or c_array instance")
    if not (isinstance(out, c_lvalue) or isinstance(out, c_array)):
        Meta_Toolbox_Error("out must be c_lvalue or c_array instance")
    if dim is None:
        dim = out.dim
    forloop, k = c_for_loop(0, dim, 1, pragma_unroll=True)
    argk = arg[k] if isinstance(arg, c_array) else arg
    outk = out[k] if isinstance(out, c_array) else out
    return forloop(outk.assign(argk))
