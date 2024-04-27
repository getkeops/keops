from .c_instruction import c_instruction_from_string
from .c_for import c_for_loop
from .c_array import c_array, c_array_from_address, c_fixed_size_array
from .misc import Meta_Toolbox_Error


def ComplexVectApply(fun, *args):
    # similar to VectApply but for complex operations
    if not all(isinstance(arg, c_array) for arg in args):
        Meta_Toolbox_Error("inputs should be c_array instances")
    dims = [arg.dim for arg in args]
    dimloop = max(dims)
    if not set(dims) in ({dimloop}, {2, dimloop}):
        Meta_Toolbox_Error("incompatible dimensions in ComplexVectApply")
    forloop, k = c_for_loop(0, dimloop, 2, pragma_unroll=True)
    argsk = [
        c_array_from_address(2, arg.c_address + (0 if arg.dim == 1 else k))
        for arg in args
    ]
    return forloop(fun(*argsk))
