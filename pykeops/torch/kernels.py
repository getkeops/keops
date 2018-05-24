import math
import re
import inspect

import torch

from pykeops.torch.utils            import Formula
from pykeops.torch.features_kernels import FeaturesKP

from pykeops.torch.utils import _squared_distances

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
kernel_formulas =  {
    "linear" :        Formula( # Linear kernel wrt. directions aka. "currents"
        formula_sum =                           "({X},{Y})",
        routine_sum = lambda xsy=None, **kwargs : xsy,
        formula_log =                      "(IntInv(2) * Log( ({X},{Y})**2 + IntInv(10000) ))",
        routine_log = lambda xsy=None, **kwargs : .5 * (xsy**2 + .0001).log()
    ),
    "gaussian" :      Formula( # Standard RBF kernel
        formula_sum = "Exp( -(WeightedSqDist({G},{X},{Y})) )",
        routine_sum = lambda gxmy2=None, **kwargs : (-gxmy2).exp(),
        formula_log = "( -(WeightedSqDist({G},{X},{Y})) )",
        routine_log = lambda gxmy2=None, **kwargs :  -gxmy2,
    ),
    "cauchy" :        Formula( # Heavy tail kernel
        formula_sum =  "Inv( IntCst(1) + WeightedSqDist({G},{X},{Y})  )",
        routine_sum = lambda gxmy2=None, **kwargs : 1. / ( 1 + gxmy2),
        formula_log =  "(  IntInv(-1) * Log(IntCst(1) + WeightedSqDist({G},{X},{Y})) ) ",
        routine_log = lambda gxmy2=None, **kwargs : -(1+gxmy2).log(),
    ),
    "laplacian" :     Formula( # Pointy kernel
        formula_sum = "Exp(-Sqrt( WeightedSqDist({G},{X},{Y}) ))",
        routine_sum = lambda gxmy2=None, **kwargs : (-(gxmy2).sqrt()).exp(),
        formula_log = "(-Sqrt( WeightedSqDist({G},{X},{Y}) ))",
        routine_log = lambda gxmy2=None, **kwargs :  -(gxmy2).sqrt(),
    ),
    "inverse_multiquadric" :  Formula( # Heavy tail kernel
        formula_sum =  "Inv(Sqrt( IntCst(1) + WeightedSqDist({G},{X},{Y}) ) )",
        routine_sum = lambda gxmy2=None, **kwargs :  torch.rsqrt( 1 + gxmy2 ),
        formula_log =  "(IntInv(-2) * Log( IntCst(1) + WeightedSqDist({G},{X},{Y}) ) ) ",
        routine_log = lambda gxmy2=None, **kwargs :   -.5 * ( 1 + gxmy2 ).log(),
    ),
}

def set_indices(formula, f_ind, v_ind) :
    """
    Modify the patterns stored in kernel_formulas, taking into account the fact that
    the current formula is the f_ind-th, working with the v_ind-th pair of variables.
    """

    # KeOps backend -------------------------------------------------------------------------
    n_params = formula.n_params
    n_vars   = formula.n_vars

    if n_params == 1 : G_str = "G_"+str(f_ind+1)
    else :             G_str = None

    if n_vars   == 2 : X_str, Y_str = "X_"+str(v_ind+1), "Y_"+str(v_ind+1)
    else :             X_str, Y_str = None, None

    formula.formula_sum = formula.formula_sum.format(G = G_str, X = X_str, Y = Y_str)
    formula.formula_log = formula.formula_log.format(G = G_str, X = X_str, Y = Y_str)

    # Vanilla PyTorch backend -------------------------------------------------------------------
    # Guess which quantities will be needed by the vanilla pytorch binding:
    params_sum = inspect.signature(formula.routine_sum).parameters
    needs_x_y_gxmy2_xsy_sum = (v_ind, 'x' in params_sum, 'y' in params_sum, 'gxmy2' in params_sum, 'xsy' in params_sum)
    formula.subroutine_sum = formula.routine_sum
    formula.routine_sum = lambda x=None, y=None, gxmy2=None, xsy=None : \
                          formula.subroutine_sum(x=x[v_ind], y=y[v_ind], gxmy2=gxmy2[f_ind], xsy=xsy[f_ind])
    
    params_log = inspect.signature(formula.routine_log).parameters
    needs_x_y_gxmy2_xsy_log = (v_ind, 'x' in params_log, 'y' in params_log, 'gxmy2' in params_log, 'xsy' in params_log)
    formula.subroutine_log = formula.routine_log
    formula.routine_log = lambda x=None, y=None, gxmy2=None, xsy=None : \
                          formula.subroutine_log(x=x[v_ind], y=y[v_ind], gxmy2=gxmy2[f_ind], xsy=xsy[f_ind])


    return formula, f_ind+n_params, needs_x_y_gxmy2_xsy_sum, needs_x_y_gxmy2_xsy_log

class Kernel :
    def __init__(self, name=None) :
        """
        Examples of valid names :
            " gaussian(x,y) * linear(u,v)**2 * gaussian(s,t)"
            " gaussian(x,y) * (1 + linear(u,v)**2 ) "
        """
        if name is not None :
            # in the comments, let's suppose that name="gaussian(x,y) + laplacian(x,y) * linear(u,v)**2"
            # Determine the features type from the formula : ------------------------------------------------
            variables = re.findall(r'(\([a-z],[a-z]\))', name) # ['(x,y)', '(x,y)', '(u,v)']
            used = set()
            variables = [x for x in variables if x not in used and (used.add(x) or True)]
            #         = ordered, "unique" list of pairs "(x,y)", "(u,v)", etc. used
            #         = ['(x,y)', '(u,v)']
            var_to_ind = { k : i for (i,k) in enumerate(variables)}
            #          = {'(x,y)': 0, '(u,v)': 1}

            subformulas_str = re.findall(r'([a-zA-Z_][a-zA-Z_0-9]*)(\([a-z],[a-z]\))', name)
            #               = [('gaussian', '(x,y)'), ('laplacian', '(x,y)'), ('linear', '(u,v)')]

            f_ind, subformulas, vars_needed_sum, vars_needed_log = 0, [], [], []
            for formula_str, var_str in subformulas_str :
                formula = kernel_formulas[formula_str] # = Formula(...)
                formula, f_ind, need_sum, need_log = set_indices(formula, f_ind, var_to_ind[var_str])
                subformulas.append(formula)
                vars_needed_sum.append(need_sum)
                vars_needed_log.append(need_log)
                
            # ...

            for (i,_) in enumerate(subformulas) :
                name = re.sub(r'[a-zA-Z_][a-zA-Z_0-9]*\([a-z],[a-z]\)',  r'subformulas[{}]'.format(i), name, count=1)

            # Replace int values "N" with "Formula(intvalue=N) (except the indices of subformulas)"
            
            name = re.sub(r'(?<!subformulas\[)([0-9]+)', r'Formula(intvalue=\1)', name)

            # Final result : ----------------------------------------------------------------------------------
            kernel = eval(name)
            
            self.formula_sum = kernel.formula_sum
            self.routine_sum = kernel.routine_sum
            self.formula_log = kernel.formula_log
            self.routine_log = kernel.routine_log
            # Two lists, needed by the vanilla torch binding
            self.routine_sum.vars_needed = vars_needed_sum
            self.routine_log.vars_needed = vars_needed_log
        else :
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

    return FeaturesKP( kernel, gamma, x, y, b, 
                       mode = mode, backend = backend, bonus_args=bonus_args)



def kernel_product(params, x,y, *bs, mode=None) :
    """
    Just a simple wrapper around the KernelProduct operation,
    with a user-friendly "dict" of parameters.
    It allows you to compute kernel dot products (aka. as discrete convolutions)
    with arbitrary formulas, using a "sum" or a "log-sum-exp" reduction operation.

    Returns: ---------------------------------------------------------------------
    - v (Variable of size (N,E)).

    If params["mode"] == "sum" (default), we have :
        v_i =     \sum_j k(x_i, y_j) b_j

    Otherwise, if params["mode"] == "lse", we have :
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
            - "mode"    : one of "sum" or "lse"
                        
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
    if mode is None : mode = params.get("mode", "sum")
    backend = params.get("backend", "auto")
    # gamma should have been generated along the lines of "Variable(torch.Tensor([1/(s**2)])).type(dtype)"
    gamma   = params["gamma"]

    if not gamma.__class__ in [tuple, list] : gamma = (gamma,)
    if not     x.__class__ in [tuple, list] :     x = (x,)
    if not     y.__class__ in [tuple, list] :     y = (y,)

    return FeaturesKP( kernel, gamma, x, y, bs, mode = mode, backend = backend)





