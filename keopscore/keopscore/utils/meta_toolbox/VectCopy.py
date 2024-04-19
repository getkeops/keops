def VectCopy(out, arg, dim=None):
    # returns a C++ code string representing a vector copy between fixed-size arrays
    # - dim is dimension of arrays
    # - out is c_variable representing the output array
    # - arg is c_variable representing the input array
    if dim is None:
        dim = out.dim
    forloop, k = c_for_loop(0, dim, 1, pragma_unroll=True)
    return forloop(out[k].assign(arg[k]))
