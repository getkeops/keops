
import torch

from .kernel_product_generic import GenericKernelProduct
from .logsumexp_generic      import GenericLogSumExp

from .utils import _scalar_products, _squared_distances, _log_sum_exp


def _locations_directions_kernel(routine, g,x,y, h,u,v, b, matrix=False) :
    """
    """
    K = routine(g=g, x=x, y=y, xmy2 = _squared_distances(x,y), \
                h=h, u=u, v=v, usv  =   _scalar_products(u,v)  )
    return K @ b if not matrix else K

def _locations_directions_kernel_log(routine, g,x,y, h,u,v, b_log, matrix=False) :
    """
    """
    C = routine(g=g, x=x, y=y, xmy2 = _squared_distances(x,y), \
                h=h, u=u, v=v, usv  =   _scalar_products(u,v)  )
    return _log_sum_exp( C + b_log.view(1,-1) , 1 ).view(-1,1) if not matrix else C

def LocationsDirectionsKP( kernel, g,x,y, h,u,v, b, mode = "sum", backend="auto") :
    """
    """
    if h is None : h = g  # Shameful HACK until I properly implement parameters for the pytorch backend!!!

    if backend == "pytorch" :
        if   mode == "sum" : return     _locations_directions_kernel(kernel.routine_sum, g,x,y, h,u,v, b)
        elif mode == "log" : return _locations_directions_kernel_log(kernel.routine_log, g,x,y, h,u,v, b)
        else : raise ValueError('"mode" should either be "sum" or "log".')
    elif backend == "matrix" :
        if   mode == "sum" : return     _locations_directions_kernel(kernel.routine_sum, g,x,y, h,u,v, b, matrix=True)
        elif mode == "log" : return _locations_directions_kernel_log(kernel.routine_log, g,x,y, h,u,v, b, matrix=True)
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
                    "H = Pm(1)"          ,   # 2nd parameter
                    "X = Vx(0,DIMPOINT)" ,   # 1st variable, dim DIM,    indexed by i
                    "Y = Vy(1,DIMPOINT)" ,   # 2nd variable, dim DIM,    indexed by j
                    "U = Vx(2,DIMPOINT)" ,   # 3rd variable, dim DIM,    indexed by i
                    "V = Vy(3,DIMPOINT)" ,   # 4th variable, dim DIM,    indexed by j
                    "B = Vy(4,DIMOUT  )" ]   # 5th variable, dim DIMOUT, indexed by j

        # stands for:     R_i   ,   G  ,       X_i    ,       Y_j    ,
        signature = [ (dimout,0), (1,2),  (dimpoint,0),  (dimpoint,1), \
        #                           H  ,       U_i    ,       V_j    ,
                                  (1,2),  (dimpoint,0),  (dimpoint,1), \
        #                              B_j    .
                                  (dimout,1) ]
        sum_index = 0 # the output vector is indexed by "i" (CAT=0)
        return genconv( backend, aliases, formula, signature, sum_index, g,x,y, h,u,v, b )
        
