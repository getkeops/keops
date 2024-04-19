from c_array import c_array


def VectApply(fun, out, *args):
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

    dims = [out.dim]
    for arg in args:
        if isinstance(arg, c_variable):
            dims.append(1)
        elif isinstance(arg, c_array):
            dims.append(arg.dim)
        else:
            KeOps_Error("args must be c_variable or c_array instances")
    dimloop = max(dims)
    if not set(dims) in ({dimloop}, {1, dimloop}):
        KeOps_Error("incompatible dimensions in VectApply")
    incr_out = 1 if out.dim == dimloop else 0
    incr_args = list((1 if dim == dimloop else 0) for dim in dims[1:])

    forloop, k = c_for_loop(0, dimloop, 1, pragma_unroll=True)

    argsk = []
    for arg, incr in zip(args, incr_args):
        if isinstance(arg, c_variable):
            argsk.append(arg)
        elif isinstance(arg, c_array):
            argsk.append(arg[k * incr])

    return forloop(fun(out[k * incr_out], *argsk))
