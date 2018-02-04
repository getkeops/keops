import math
import re

import torch

from .utils            import Formula
from .features_kernels import FeaturesKP


# Define the standard kernel building blocks. 
# They will be concatenated depending on the "name" argument of Kernel.__init__
# Feel free to add your own "pet formula" at run-time, 
# using for instance :
#  " kernels.locations_formulas["mykernel"] = utils.Formula( ... ) "

# The "formula_*" attributes are the ones that will be used by the
# C++/CUDA "libkp" routines - backend == "auto", "CPU", "GPU_1D", "GPU_2D".
# They can rely on the following aliases :
# "G", "X", "Y",
# "H", "U", "V",
# "I", "S", "T"

# The "routine_*" ones are used if backend == "pytorch",
# and allow you to check independently the correctness of the C++ code.
# These pytorch routines can make use of
# "g", "x", "y" and "xmy2" (= |x-y|_2^2),
# "h", "u", "v" and "usv"  (= <u,v>_2  ),
# "i", "s", "t" and "smt2" (= |s-t|_2^2).

# Formulas in "x_i" and "y_j", with parameters "g" (=1/sigma^2, for instance)
locations_formulas = {
    "gaussian" :      Formula( # Standard RBF kernel
        formula_sum =                          "Exp( -(Cst(G) * SqDist(X,Y)) )",
        routine_sum = lambda g=None, xmy2=None, **kwargs : (-g*xmy2).exp(),
        formula_log =                             "( -(Cst(G) * SqDist(X,Y)) )",
        routine_log = lambda g=None, xmy2=None, **kwargs :  -g*xmy2,
    ),
    "exponential" :   Formula( # Pointy kernel
        formula_sum =                      "Exp( - Sqrt(Cst(G) * SqDist(X,Y)  + IntInv(10000) ) )",
        routine_sum = lambda g=None, xmy2=None, **kwargs : (-(g*xmy2+.0001).sqrt()).exp(),
        formula_log =                         "(  - Sqrt(Cst(G) * SqDist(X,Y) + IntInv(10000) ) )",
        routine_log = lambda g=None, xmy2=None, **kwargs :  -(g*xmy2+.0001).sqrt(),
    ),
    "energy" :        Formula( # Heavy tail kernel
        formula_sum =   "Powf( IntCst(1) + Cst(G) * SqDist(X,Y) , IntInv(-4) )",
        routine_sum = lambda g=None, xmy2=None, **kwargs : torch.pow( 1 + g * xmy2, -.25 ),
        formula_log =       "(  IntInv(-4) * Log(IntCst(1) + Cst(G) * SqDist(X,Y)) ) ",
        routine_log = lambda g=None, xmy2=None, **kwargs :  -.25 * (1 + g * xmy2).log(),
    ),
}

# Formulas in "u_i" and "v_j", with parameters "h" (=1/sigma^2, for instance)
directions_formulas = {
    "linear" :        Formula( # Linear kernel wrt. directions aka. "currents"
        formula_sum =                           "(U,V)",
        routine_sum = lambda usv=None, **kwargs : usv,
        formula_log =                      "(IntInv(2) * Log( (U,V)**2 + IntInv(10000) ))",
        routine_log = lambda usv=None, **kwargs : .5 * (usv**2 + .0001).log()
    ),
}

# Formulas in "s_i" and "t_j", with parameters "i" (=1/sigma^2, for instance)
values_formulas = {
    "gaussian" :    Formula( # Standard RBF kernel
        formula_sum =                          "Exp( -(Cst(I) * SqDist(S,T)) )",
        routine_sum = lambda i=None, smt2=None, **kwargs : (-i*smt2).exp(),
        formula_log =                             "( -(Cst(I) * SqDist(S,T)) )",
        routine_log = lambda i=None, smt2=None, **kwargs :  -i*smt2,
    ),
}


class Kernel :
    def __init__(self, name=None) :
        """
        Examples of valid names :
            " gaussian(x,y) * linear(u,v)**2 * gaussian(s,t)"
            " gaussian(x,y) * (1 + linear(u,v)**2 ) "
        """
        if name is not None :
            # Determine the features type from the formula : ------------------------------------------------
            locations  = "(x,y)" in name
            directions = "(u,v)" in name
            values     = "(s,t)" in name

            if   locations and not directions and not values :
                self.features    = "locations"
            elif locations and     directions and not values :
                self.features    = "locations+directions"
            elif locations and     directions and     values :
                self.features    = "locations+directions+values"
            else :
                raise ValueError( "This combination of features is not supported (yet) : \n" \
                                + "locations : "+str(locations) + ", directions : " + str(directions) \
                                + ", values : " + str(values) +".")

            # Regexp matching ---------------------------------------------------------------------------------
            # Replace, say, " gaussian(x,y) " with " locations_formulas["gaussian"] "
            name = re.sub(r'([a-zA-Z_][a-zA-Z_0-9]*)\(x,y\)',  r'locations_formulas["\1"]', name)
            name = re.sub(r'([a-zA-Z_][a-zA-Z_0-9]*)\(u,v\)', r'directions_formulas["\1"]', name)
            name = re.sub(r'([a-zA-Z_][a-zA-Z_0-9]*)\(s,t\)',     r'values_formulas["\1"]', name)
            # Replace int values "N" with "Formula(intvalue=N)"
            name = re.sub(r'([0-9]+)',     r'Formula(intvalue=\1)', name)

            # Final result : ----------------------------------------------------------------------------------
            kernel = eval(name)
            
            self.formula_sum = kernel.formula_sum
            self.routine_sum = kernel.routine_sum
            self.formula_log = kernel.formula_log
            self.routine_log = kernel.routine_log
        else :
            self.features    = None
            self.formula_sum = None
            self.routine_sum = None
            self.formula_log = None
            self.routine_log = None



def KernelProduct(gamma, x,y,b, kernel, mode, backend = "auto", bonus_args=None) :
    """
    Convenience function.

    Computes K(x_i,y_j) @ nu_j = \sum_j k(x_i-y_j) * nu_j
    where k is a kernel function specified by "kernel", a 'Kernel' object.

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

    However, if, say, kernel.features = "locations+directions",
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

    if   kernel.features == "locations" :
        G     = gamma; X     = x; Y     = y
        return FeaturesKP( kernel, G,X,Y,               b, 
                        mode = mode, backend = backend, bonus_args=bonus_args)
                        
    elif kernel.features == "locations+directions" :
        G,H   = gamma; X,U   = x; Y,V   = y
        return FeaturesKP( kernel, G,X,Y, H,U,V,        b,
                        mode = mode, backend = backend, bonus_args=bonus_args)

    elif kernel.features == "locations+directions+values" :
        G,H,I = gamma; X,U,S = x; Y,V,T = y
        return FeaturesKP( kernel, G,X,Y, H,U,V, I,S,T, b, 
                        mode = mode, backend = backend, bonus_args=bonus_args)
    else :
        raise NotImplementedError("Kernel features '"+kernel.features+"'. "\
                                +'Available values are "locations" (measures), "locations+directions" (shapes)' \
                                +'and "locations+directions+values" (fshapes).' )



def kernel_product(x,y,b, params, mode = "sum", bonus_args = None) :
    """
    Just a simple wrapper around the KernelProduct operation,
    with a user-friendly "dict" of parameters.
    It allows you to compute kernel dot products (aka. as discrete convolutions)
    with arbitrary formulas, using a "sum" or a "log-sum-exp" reduction operation.

    Returns: ---------------------------------------------------------------------
    - v (Variable of size (N,E)).

    If mode == "sum", we have :
        v_i =     \sum_j k(x_i, y_j) b_j

    Otherwise, if mode == "log", we have :
        v_i = log \sum_j exp( c(x_i, y_j) + b_j )
    where c(x_i,y_j) = log( k(x_i,y_j) )  -- computed with improved numerical accuracy.

    Args: -------------------------------------------------------------------------
    - x   (Variable, or a F-tuple of Variables) : 
            The "F" features of the points (x_i), 1 <= i <= N.
            All the Variables should be 2d-tensors, with the same size "N"
            along dimension 0.
    - y   (Variable, or a F-tuple of Variables) : 
            The "F" features of the points (y_j), 1 <= j <= M.
            All the Variables should be 2d-tensors, with the same size "M"
            along dimension 0.
    - b   (Variable of size (M,E)) :
            The vectors associated to the points y_j.
    - params :
            A dictionnary, which describes the kernel being used.
            It should have the following attributes :
            - "id"      : a libkp.torch.kernels.Kernel object,
                        which describes the formulas for "k", "c" and the number of features
                        ("locations", "location+directions", etc.) being used.
            - "backend" : "auto",    to use libkp's CPU or CUDA routines (default option).
                        "pytorch", to fall back on a reference matrix implementation.
            - "gamma"   : a F-tuple of scalar Variables.
                        Typically, something along the lines of
                        "(1/sigma**2, 1/tau**2, 1/kappa**2)" ...
                        
    Typical examples of use are given in the tutorials.


    BONUS MODES : -----------------------------------------------------------------
    on top of the "normal" kernel products, we provide additionnal
    operations, related to the Sinkhorn scaling algorithm.
    These operations require two additional parameters,
    referred to as "bonus_args = (Alog,Blog)",
    Variables of size (N,1) and (M,1) respectively,
    which encode the logarithms of the scaling coefficients.

    If       mode == "log_scaled", we have :
        v_i =     \sum_j exp( c(x_i,y_j) + Alog_i + Blog_j ) * b_j

    Else, if mode == "log_scaled_log", we have :
        v_i = log \sum_j exp( c(x_i,y_j) + Alog_i + Blog_j + b_j )

    Else, if mode == "log_primal", we have :
        v_i = \sum_j (Alog_i+Blog_j-1) * exp( c(x_i,y_j) + Alog_i + Blog_j )
        (b_j is not used)

    Else, if mode == "log_cost", we have :
        v_i = \sum_j -c(x_i,y_j) * exp( c(x_i,y_j) + Alog_i + Blog_j )
        (b_j is not used)
    """
    kernel  = params["id"]
    backend = params.get("backend", "auto")
    # gamma should have been generated along the lines of "Variable(torch.Tensor([1/(s**2)])).type(dtype)"
    gamma   = params["gamma"]
    
    return KernelProduct(gamma, x,y,b, kernel, mode, backend, bonus_args = bonus_args)





