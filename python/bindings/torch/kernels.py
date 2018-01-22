#import os.path
#import sys
#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '')

from .kernel_product_generic import GenericKernelProduct
from .logsumexp_generic import GenericLogSumExp

def _squared_distances(x, y) :
    x_i = x.unsqueeze(1)         # Shape (N,D) -> Shape (N,1,D)
    y_j = y.unsqueeze(0)         # Shape (M,D) -> Shape (1,M,D)
    return ((x_i-y_j)**2).sum(2) # N-by-M matrix, xmy[i,j] = |x_i-y_j|^2

def _radial_kernel(routine, x, y, b) :
    K = routine(_squared_distances(x,y))
    return K @ b  # Matrix product between the Kernel operator and the source field b

def _log_sum_exp(mat, dim):
    """
    Computes the log-sum-exp of a matrix with a numerically stable scheme, 
    in the user-defined summation dimension: exp is never applied
    to a number >= 0, and in each summation row, there is at least
    one "exp(0)" to stabilize the sum.
    
    For instance, if dim = 1 and mat is a 2d array, we output
                log( sum_j exp( mat[i,j] )) 
    by factoring out the row-wise maximas.
    """
    max_rc = torch.max(mat, dim)[0]
    return max_rc + torch.log(torch.sum(torch.exp(mat - max_rc.unsqueeze(dim)), dim))

def _radial_kernel_log(routine, x, y, b_log) :
    """
    Computes log( K(x_i,y_j) @ b_j) = log( \sum_j k(x_i-y_j) * b_j) in the log domain,
    where k is a kernel function speciefied by "routine".
    """
    C = routine(_squared_distances(x,y))
    return _log_sum_exp( C + b_log.view(1,-1) , 1 ).view(-1,1) 

def RadialKernel( formula, routine, gamma, x, y, b, mode = "sum", backend="auto") :
    if backend == "pytorch" :
        if   mode == "sum" : return     _radial_kernel(routine, x, y, b)
        elif mode == "log" : return _radial_kernel_log(routine, x, y, b)
        else : raise ValueError('"mode" should either be "sum" or "log".')

    else :
        if   mode == "sum" : 
            genconv  = GenericKernelProduct().apply
            formula += " * B"
        elif mode == "log" :
            genconv  = GenericLogSumExp().apply
            formula += " + B"
        else : raise ValueError('"mode" should either be "sum" or "log".')
        
        dimpoint = x.size(1) ; dimout = b.size(1)
        
        aliases  = ["DIMPOINT = "+str(dimpoint), "DIMOUT = "+str(dimout),
                    "G = Pm(0)"          ,   # 1st parameter
                    "X = Vx(0,DIMPOINT)" ,   # 1st variable, dim DIM,    indexed by i
                    "Y = Vy(1,DIMPOINT)" ,   # 2nd variable, dim DIM,    indexed by j
                    "B = Vy(2,DIMOUT  )" ]   # 3rd variable, dim DIMOUT, indexed by j

        # stands for:     R_i   ,   G  ,      X_i    ,      Y_j    ,     B_j    .
        signature = [ (dimout,0), (1,2), (dimpoint,0), (dimpoint,1), (dimout,1) ]
        sum_index = 0 # the output vector is indexed by "i" (CAT=0)
        return genconv( backend, aliases, formula, signature, sum_index, gamma, x, y, b )
    

def GaussianKernel( gamma, x, y, b, mode = "sum", backend = "auto") :
    if   mode=="sum": 
        formula = "Exp( -(Cst(G) * SqNorm2(X-Y)) )"
        routine = lambda xmy2 : (-gamma*xmy2).exp()
    elif mode=="log": 
        formula =    "( -(Cst(G) * SqNorm2(X-Y)) )"
        routine = lambda xmy2 :  -gamma*xmy2
    else : raise ValueError('"mode" should either be "sum" or "log".')
    return RadialKernel( formula, routine, gamma, x, y, b, mode, backend)


"""
    elif mode == "laplace"  : K = torch.exp( - torch.sqrt(xmy + (s**2)) )
    elif mode == "energy"   : K = torch.pow(   xmy + (s**2), -.25 )

    if   mode == "gaussian"     : C =  - xmy / (s**2) 
    elif mode == "laplace"      : C =  - torch.sqrt(xmy + (s**2)) 
    elif mode == "energy"       : C =  -.25 * torch.log( xmy + (s**2) )
    elif mode == "exponential"  : C =  - torch.sqrt(xmy) / s 
"""


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

    point_kernels           = { "gaussian"         : GaussianKernel }
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














