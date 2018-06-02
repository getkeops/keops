import math
import re
import inspect
import copy

import torch

from pykeops.torch.utils            import Formula
from pykeops.torch.features_kernels import FeaturesKP

from pykeops.torch.utils import _squared_distances

# Define the standard kernel building blocks. 
# They will be concatenated depending on the "name" argument of Kernel.__init__
# Feel free to add your own "pet formula" at run-time, 
# using for instance :
#  " kernel_formulas["mykernel"] = Formula( ... ) "

# In some cases, due to PyTorch's behavior mainly, we have to add a small
# epsilon in front of square roots and logs. As for KeOps code,
# note that [dSqrt(x)/dx](x=0) has been conventionally set to 0.
Epsilon = "IntInv(100000000)"
epsilon = 1e-8

# Formulas in "x_i" and "y_j", with parameters "g" (=1/sigma^2, for instance)
kernel_formulas =  {
    "linear" :        Formula( # Linear kernel
        formula_sum =                           "({X},{Y})",
        routine_sum = lambda xsy=None, **kwargs : xsy,
        formula_log =                      "(IntInv(2) * Log( ({X},{Y})**2 + "+Epsilon+" ))",
        routine_log = lambda xsy=None, **kwargs : .5 * (xsy**2 + epsilon).log()
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
        routine_sum = lambda gxmy2=None, **kwargs : (-(gxmy2+epsilon).sqrt()).exp(),
        formula_log = "(-Sqrt( WeightedSqDist({G},{X},{Y}) ))",
        routine_log = lambda gxmy2=None, **kwargs :  -(gxmy2+epsilon).sqrt(),
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

    if n_params == 1 : G_str = "G_"+str(f_ind)
    else :             G_str = None

    if n_vars   == 2 : X_str, Y_str = "X_"+str(v_ind), "Y_"+str(v_ind)
    else :             X_str, Y_str = None, None

    formula.formula_sum = formula.formula_sum.format(G = G_str, X = X_str, Y = Y_str)
    formula.formula_log = formula.formula_log.format(G = G_str, X = X_str, Y = Y_str)

    # Vanilla PyTorch backend -------------------------------------------------------------------
    # Guess which quantities will be needed by the vanilla pytorch binding:
    params_sum = inspect.signature(formula.routine_sum).parameters
    needs_x_y_gxmy2_xsy_sum = (v_ind, 'x' in params_sum, 'y' in params_sum, 'gxmy2' in params_sum, 'xsy' in params_sum)
    formula.subroutine_sum = formula.routine_sum
    formula.routine_sum = lambda x=None, y=None, g=None, gxmy2=None, xsy=None : \
                          formula.subroutine_sum(x=x[v_ind], y=y[v_ind], gxmy2=gxmy2[f_ind], xsy=xsy[f_ind])
    
    params_log = inspect.signature(formula.routine_log).parameters
    needs_x_y_gxmy2_xsy_log = (v_ind, 'x' in params_log, 'y' in params_log, 'gxmy2' in params_log, 'xsy' in params_log)
    formula.subroutine_log = formula.routine_log
    formula.routine_log = lambda x=None, y=None, g=None, gxmy2=None, xsy=None : \
                          formula.subroutine_log(x=x[v_ind], y=y[v_ind], gxmy2=gxmy2[f_ind], xsy=xsy[f_ind])


    return formula, f_ind+1, needs_x_y_gxmy2_xsy_sum, needs_x_y_gxmy2_xsy_log

class Kernel :
    def __init__(self, name = None, formula_sum=None, routine_sum=None,
                                    formula_log=None, routine_log=None ) :
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

            # f_ind = index of the current formula
            # subformulas = list of formulas used in the kernel_product
            # vars_needed_sum and vars_needed_log keep in mind the symbolic pre-computations
            # |x-y|^2 and <x,y> that may be needed by the Vanilla PyTorch backend.
            f_ind, subformulas, vars_needed_sum, vars_needed_log = 0, [], [], []
            for formula_str, var_str in subformulas_str :
                # Don't forget the copy! This code should have no side effect on kernel_formulas!
                formula = copy.copy( kernel_formulas[formula_str] ) # = Formula(...)
                # Modify the symbolic "formula" to let it take into account the formula and variable indices:
                formula, f_ind, need_sum, need_log = set_indices(formula, f_ind, var_to_ind[var_str])
                # Store everyone for later use and substitution:
                subformulas.append(formula)
                vars_needed_sum.append(need_sum)
                vars_needed_log.append(need_log)
                
            # One after another, replace the symbolic "name(x,y)" by references to our list of "index-aware" formulas
            for (i,_) in enumerate(subformulas) :
                name = re.sub(r'[a-zA-Z_][a-zA-Z_0-9]*\([a-z],[a-z]\)',  r'subformulas[{}]'.format(i), name, count=1)
            #        = "subformulas[0] + subformulas[1] * subformulas[2]**2"

            # Replace int values "N" with "Formula(intvalue=N)"" (except the indices of subformulas!)
            name = re.sub(r'(?<!subformulas\[)([0-9]+)', r'Formula(intvalue=\1)', name)

            # Final result : ----------------------------------------------------------------------------------
            kernel = eval(name) # It's a bit dirty... Please forgive me !
            
            # Store the required info
            self.formula_sum = kernel.formula_sum
            self.routine_sum = kernel.routine_sum
            self.formula_log = kernel.formula_log
            self.routine_log = kernel.routine_log
            # Two lists, needed by the vanilla torch binding
            self.routine_sum.vars_needed = vars_needed_sum
            self.routine_log.vars_needed = vars_needed_log

        else :
            self.formula_sum = formula_sum
            self.routine_sum = routine_sum
            self.formula_log = formula_log
            self.routine_log = routine_log


def kernel_product(params, x,y, *bs, mode=None) :
    r"""
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





