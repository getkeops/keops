
import torch

from .kernel_product_generic import GenericKernelProduct
from .logsumexp_generic      import GenericLogSumExp

from .utils import _squared_distances, _log_sum_exp

def _scalar_radial_kernel(routine, x, y, b) :
    """
    Computes (\sum_j k(x_i-y_j) * b_j)_i as a simple matrix product,
    where k is a kernel function specified by "routine".
    """
    K = routine(_squared_distances(x,y))
    return K @ b  # Matrix product between the Kernel operator and the source field b

def _scalar_radial_kernel_log(routine, x, y, b_log) :
    """
    Computes log( K(x_i,y_j) @ b_j) = log( \sum_j k(x_i-y_j) * b_j) in the log domain,
    where k is a kernel function specified by "routine".
    """
    C = routine(_squared_distances(x,y))
    return _log_sum_exp( C + b_log.view(1,-1) , 1 ).view(-1,1) 

def ScalarRadialKernel( formula, routine, gamma, x, y, b, mode = "sum", backend="auto") :
    """
    Convenience function for .
    It 
    """

    if backend == "pytorch" :
        if   mode == "sum" : return     _scalar_radial_kernel(routine, x, y, b)
        elif mode == "log" : return _scalar_radial_kernel_log(routine, x, y, b)
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
    """
    The standard gaussian/RBF kernel, whose "log" is no one but the squared L2 norm:
                        k(x,y) = exp( - gamma * |x-y|^2 ) .
    """
    if   mode=="sum": 
        formula = "Exp( -(Cst(G) * SqNorm2(X-Y)) )"
        routine = lambda xmy2 : (-gamma*xmy2).exp()
    elif mode=="log": 
        formula =    "( -(Cst(G) * SqNorm2(X-Y)) )"
        routine = lambda xmy2 :  -gamma*xmy2
    else : raise ValueError('"mode" should either be "sum" or "log".')
    return ScalarRadialKernel( formula, routine, gamma, x, y, b, mode, backend)

def ExponentialKernel( gamma, x, y, b, mode = "sum", backend = "auto") :
    """
    The pointy "exponential" kernel, whose log is given by the norm:
                k(x,y) = exp( - sqrt(gamma*|x-y|^2) ) .
    """
    if   mode=="sum": 
        formula =   "Exp( - Sqrt(Cst(G) * SqNorm2(X-Y)) )"
        routine = lambda xmy2 : (-(gamma*xmy2).sqrt()).exp()
    elif mode=="log": 
        formula =      "( - Sqrt(Cst(G) * SqNorm2(X-Y)) )"
        routine = lambda xmy2 :  -(gamma*xmy2).sqrt()
    else : raise ValueError('"mode" should either be "sum" or "log".')
    return ScalarRadialKernel( formula, routine, gamma, x, y, b, mode, backend)

def EnergyKernel( gamma, x, y, b, mode = "sum", backend = "auto") :
    """
    The heavy-tail "energy" kernel :   
                k(x,y) = 1 / ( 1 + gamma*|x-y|^2 )^.25   ~   1 / sqrt(|x-y|) .
    """
    if   mode=="sum": 
        formula = "Powf( IntCst(1) + Cst(G) * SqNorm2(X-Y) , IntInv(-4) )"
        routine = lambda xmy2 : torch.pow( 1 + gamma * xmy2, -.25 )
    elif mode=="log": 
        formula =       "( IntInv(-4) * Log(IntCst(1) + Cst(G) * SqNorm2(X-Y)) )"
        routine = lambda xmy2 :  -.25 * (1 + gamma * xmy2).log()
    else : raise ValueError('"mode" should either be "sum" or "log".')
    return ScalarRadialKernel( formula, routine, gamma, x, y, b, mode, backend)
