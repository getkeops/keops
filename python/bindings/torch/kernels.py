from .scalar_radial_kernels import GaussianKernel, EnergyKernel

def StandardKernelProduct(gamma, x,y,b, name, mode, backend = "auto") :
    """
    Convenience function, providing the standard formulas implemented
    in the libkp library.

    Computes K(x_i,y_j) @ nu_j = \sum_j k(x_i-y_j) * nu_j
    where k is a kernel function specified by "name".

    In the simplest cases, the signature of our function is as follow:
    Args:
        x ( (N,D) torch Variable) : sampling point cloud 
        y ( (M,D) torch Variable) : source point cloud
        b ( (M,E) torch Variable) : source vector field, supported by y
        params (dict)			  : convenient way of storing the kernel's parameters
        mode   (string)			  : either "sum" (for classical summation) 
                                    or "log" (outputs the 'log' of "sum", computed in a
                                    numerically robust way)

    Returns:
        v ( (N,E) torch Variable) : sampled vector field 
                                    (or its coordinate-wise logarithm, if mode=="log")

    However, if, say, params["formula"]=="gaussian currents",
    we use a kernel function parametrized by locations
    X_i, Y_j AND directions U_i, V_j.
    The argument "x" is then a pair (X,U) of (N,D) torch Variables,
    while        "y" is      a pair (Y,V) of (M,E) torch Variables.

    Possible values for "name" are:
        - "gaussian"

    N.B.: The backend is specified in params["backend"], and its value
        may have a critical influence on performance:
    - "pytorch" means that we use a naive "matrix-like", full-pytorch formula.
                It does not scale well as soon as N or M > 5,000.
    - "CPU"     means that we use a CPU implementation of the libkp C++ routines.
                It performs okay-ish, but cannot rival GPU implementations.
    - "GPU_2D"  means that we use a GPU implementation of the libkp C++/CUDA routines,
                with a 2-dimensional job distribution scheme. 
                It may come useful if, for instance, N < 200 and 10,000 < M. 
    - "GPU_1D"  means that we use a GPU implementation of the libkp C++/CUDA routines,
                with a 1-dimensional distribution scheme (one thread = one line of x).
                If you own an Nvidia GPU, this is the go-to method for large point clouds. 

    If the backend is not specified or "auto", the libkp routines will try to
    use a suitable one depending on your configuration + the dimensions of x, y and b.
    """

    point_kernels           = { "gaussian"         : GaussianKernel,
                                "energy"           : EnergyKernel  }
    point_direction_kernels = { }#"gaussian current" : GaussianCurrentKernel} 

    if   name in point_kernels :
        return point_kernels[name]( gamma, x, y, b, mode = mode, backend = backend)
    elif name in point_direction_kernels :
        X,U = x; Y,V = y
        return point_direction_kernels[name]( gamma, X, Y, U, V, b, mode = mode, backend = backend)
    else :
        raise NotImplementedError("Kernel name '"+name+"'. "\
                                 +'Available values are "' + '", "'.join(point_kernels.keys()) \
                                 + '", "' + '", "'.join(point_direction_kernels.keys())+'".' )














