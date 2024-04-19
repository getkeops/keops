from .c_array import c_array


def ComplexVectApply(fun, out, *args):
    # similar to VectApply but for complex operations

    dims = [out.dim]
    for arg in args:
        if isinstance(arg, c_array):
            dims.append(arg.dim)
        else:
            KeOps_Error("args must be c_array instances")
    dimloop = max(dims)
    if not set(dims) in ({dimloop}, {2, dimloop}):
        KeOps_Error("incompatible dimensions in ComplexVectApply")
    incr_out = 1 if out.dim == dimloop else 0
    incr_args = list((1 if dim == dimloop else 0) for dim in dims[1:])

    forloop, k = c_for_loop(0, dimloop, 2, pragma_unroll=True)

    argsk = []
    for arg, incr in zip(args, incr_args):
        argk = c_array(arg.dtype, 2, f"({arg.id}+{k.id}*{incr})")
        argsk.append(argk)
    outk = c_array(out.dtype, 2, f"({out.id}+{k.id}*{incr_out})")
    return forloop(fun(outk, *argsk))
