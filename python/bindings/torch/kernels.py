import math
import re

from .utils                               import Formula
from .locations_kernels                   import LocationsKP
from .locations_directions_kernels        import LocationsDirectionsKP
from .locations_directions_values_kernels import LocationsDirectionsValuesKP


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
        formula_sum =                      "Exp( - Sqrt(Cst(G) * SqDist(X,Y)) )",
        routine_sum = lambda g=None, xmy2=None, **kwargs : (-(g*xmy2).sqrt()).exp(),
        formula_log =                         "(  - Sqrt(Cst(G) * SqDist(X,Y)) )",
        routine_log = lambda g=None, xmy2=None, **kwargs :  -(g*xmy2).sqrt(),
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
        formula_log =                      "Log( (U,V) )",
        routine_log = lambda usv=None, **kwargs : usv.log()
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
    def __init__(self, name) :
        """
        Examples of valid names :
            " gaussian(x,y) * linear(u,v)**2 * gaussian(s,t)"
            " gaussian(x,y) * (1 + linear(u,v)**2 ) "
        """
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

        print(name)
        # Final result : ----------------------------------------------------------------------------------
        kernel = eval(name)
        
        self.formula_sum = kernel.formula_sum
        self.routine_sum = kernel.routine_sum
        self.formula_log = kernel.formula_log
        self.routine_log = kernel.routine_log

        print(self.formula_sum)


def KernelProduct(gamma, x,y,b, kernel, mode, backend = "auto") :
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
        return                 LocationsKP( kernel, gamma, x, y,         b, mode = mode, backend = backend)
    elif kernel.features == "locations+directions" :
        G,H   = gamma; X,U   = x; Y,V   = y
        return       LocationsDirectionsKP( kernel, G,X,Y, H,U,V,        b, mode = mode, backend = backend)
    elif kernel.features == "locations+directions+values" :
        G,H,I = gamma; X,U,S = x; Y,V,T = y
        return LocationsDirectionsValuesKP( kernel, G,X,Y, H,U,V, I,S,T, b, mode = mode, backend = backend)
    else :
        raise NotImplementedError("Kernel features '"+kernel.features+"'. "\
                                 +'Available values are "locations" (measures), "locations+directions" (shapes)' \
                                 +'and "locations+directions+values" (fshapes).' )














