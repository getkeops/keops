
import torch

from .kernel_product_generic import GenericKernelProduct
from .logsumexp_generic      import GenericLogSumExp

from .utils import _squared_distances, _log_sum_exp

def _locations_kernel(routine, g,x,y, b, matrix=False) :
    """
    Computes (\sum_j k(x_i-y_j) * b_j)_i as a simple matrix product,
    where k is a kernel function specified by "routine".
    """
    K = routine(g=g, x=x, y=y, xmy2 = _squared_distances(x,y))
    return K @ b  if not matrix else K

def _locations_kernel_log(routine, g,x,y, b_log, matrix=False) :
    """
    Computes log( K(x_i,y_j) @ b_j) = log( \sum_j k(x_i-y_j) * b_j) in the log domain,
    where k is a kernel function specified by "routine".
    """
    C = routine(g=g, x=x, y=y, xmy2 = _squared_distances(x,y))
    return _log_sum_exp( C + b_log.view(1,-1) , 1 ).view(-1,1) if not matrix else C

def LocationsKP( kernel, g,x,y, b, mode = "sum", backend="auto") :
    """
    """

    if backend == "pytorch" :
        if   mode == "sum" : return     _locations_kernel(kernel.routine_sum, g, x, y, b)
        elif mode == "log" : return _locations_kernel_log(kernel.routine_log, g, x, y, b)
        else : raise ValueError('"mode" should either be "sum" or "log".')
    elif backend == "matrix" : # Provided for convenience in visualization codes, etc: output K instead of K@b
        if   mode == "sum" : return     _locations_kernel(kernel.routine_sum, g, x, y, b, matrix=True)
        elif mode == "log" : return _locations_kernel_log(kernel.routine_log, g, x, y, b, matrix=True)
        else : raise ValueError('"mode" should either be "sum" or "log".')

    else :
        if   mode == "sum" : 
            genconv  = GenericKernelProduct().apply
            formula  = "("+kernel.formula_sum + " * B)"
        elif mode == "log" :
            genconv  = GenericLogSumExp().apply
            formula  = "("+kernel.formula_log + " + B)"
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
        return genconv( backend, aliases, formula, signature, sum_index, g, x, y, b )
    
