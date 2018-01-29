
import torch

from .kernel_product_generic import GenericKernelProduct
from .logsumexp_generic      import GenericLogSumExp

from .utils import _scalar_products, _squared_distances, _log_sum_exp


def _features_kernel(features, routine, *args, matrix=False) :
    """
    """
    if   features == "locations" :
        g,x,y, b = args
        K = routine(g=g, x=x, y=y, xmy2 = _squared_distances(x,y))

    elif features == "locations+directions" :
        g,x,y, h,u,v, b = args
        K = routine(g=g, x=x, y=y, xmy2 = _squared_distances(x,y), \
                    h=h, u=u, v=v, usv  =   _scalar_products(u,v)  )

    elif features == "locations+directions+values" :
        g,x,y, h,u,v, i,s,t, b = args
        K = routine(g=g, x=x, y=y, xmy2 = _squared_distances(x,y), \
                    h=h, u=u, v=v, usv  =   _scalar_products(u,v), \
                    i=i, s=s, t=t, smt2 = _squared_distances(s,t)  )
    else :
        raise ValueError("This number of arguments is not supported!")

    return K if matrix else K @ b


def _features_kernel_log(features, routine, *args, matrix=False) :
    """
    """
    if   features == "locations" :
        g,x,y, b_log = args
        K_log = routine(g=g, x=x, y=y, xmy2 = _squared_distances(x,y))

    elif features == "locations+directions" :
        g,x,y, h,u,v, b_log = args
        K_log = routine(g=g, x=x, y=y, xmy2 = _squared_distances(x,y), \
                        h=h, u=u, v=v, usv  =   _scalar_products(u,v)  )

    elif features == "locations+directions+values" :
        g,x,y, h,u,v, i,s,t, b_log = args
        K_log = routine(g=g, x=x, y=y, xmy2 = _squared_distances(x,y), \
                        h=h, u=u, v=v, usv  =   _scalar_products(u,v), \
                        i=i, s=s, t=t, smt2 = _squared_distances(s,t)  )
    else :
        raise ValueError("This number of arguments is not supported!")

    return K_log if matrix else _log_sum_exp( K_log + b_log.view(1,-1) , 1 ).view(-1,1)

def _features_kernel_log_scaled(features, routine, *args, matrix=False) :
    a_log, b_log = args[-2:] # scaling coefficients, typically given as output of the Sinkhorn loop
    K_log        = _features_kernel_log(features, routine, *args[:-2], matrix=True)
    aKb_log      = (a_log.view(-1,1) + b_log.view(1,-1)) + K_log
    return aKb_log if matrix else aKb_log.exp() @ args[-3]

def _features_kernel_log_scaled_log(features, routine, *args, matrix=False) :
    a_log, b_log = args[-2:] # scaling coefficients, typically given as output of the Sinkhorn loop
    K_log        = _features_kernel_log(features, routine, *args[:-2], matrix=True)
    aKb_log      = (a_log.view(-1,1) + b_log.view(1,-1)) + K_log
    return aKb_log if matrix else _log_sum_exp( aKb_log + args[-3].view(1,-1) , 1 ).view(-1,1)

def _features_kernel_log_primal(features, routine, *args, matrix=False) :
    a_log, b_log = args[-2:] # scaling coefficients, typically given as output of the Sinkhorn loop
    K_log        = _features_kernel_log(features, routine, *args[:-2], matrix=True)
    aKb_log      = (a_log.view(-1,1) + b_log.view(1,-1)) + K_log
    minus_CGamma = (a_log.view(-1,1) + b_log.view(1,-1) - 1.) * aKb_log.exp()
    return minus_CGamma if matrix else minus_CGamma.sum(1).view(-1,1)

def _features_kernel_log_cost(features, routine, *args, matrix=False) :
    a_log, b_log = args[-2:] # scaling coefficients, typically given as output of the Sinkhorn loop
    K_log        = _features_kernel_log(features, routine, *args[:-2], matrix=True)
    aKb_log      = (a_log.view(-1,1) + b_log.view(1,-1)) + K_log
    minus_CGamma = - K_log * aKb_log.exp()
    return minus_CGamma if matrix else minus_CGamma.sum(1).view(-1,1)


def FeaturesKP( kernel, *args, mode = "sum", backend="auto", bonus_args=None) :
    """
    *args = g,x,y, h,u,v, i,s,t, b
    """
    if bonus_args is not None :     args += tuple(bonus_args)

    if backend == "pytorch" :
        if   mode == "sum"            : return _features_kernel(                kernel.features, kernel.routine_sum, *args)
        elif mode == "log"            : return _features_kernel_log(            kernel.features, kernel.routine_log, *args)
        elif mode == "log_scaled"     : return _features_kernel_log_scaled(     kernel.features, kernel.routine_log, *args)
        elif mode == "log_scaled_log" : return _features_kernel_log_scaled_log( kernel.features, kernel.routine_log, *args)
        elif mode == "log_primal"     : return _features_kernel_log_primal(     kernel.features, kernel.routine_log, *args)
        elif mode == "log_cost"       : return _features_kernel_log_cost(       kernel.features, kernel.routine_log, *args)
        else : raise ValueError('"mode" should either be "sum" or "log".')
    elif backend == "matrix" :
        if   mode == "sum"            : return _features_kernel(                kernel.features, kernel.routine_sum, *args, matrix=True)
        elif mode == "log"            : return _features_kernel_log(            kernel.features, kernel.routine_log, *args, matrix=True)
        elif mode == "log_scaled"     : return _features_kernel_log_scaled(     kernel.features, kernel.routine_log, *args, matrix=True)
        elif mode == "log_scaled_log" : return _features_kernel_log_scaled_log( kernel.features, kernel.routine_log, *args, matrix=True)
        elif mode == "log_primal"     : return _features_kernel_log_primal(     kernel.features, kernel.routine_log, *args, matrix=True)
        elif mode == "log_cost"       : return _features_kernel_log_cost(       kernel.features, kernel.routine_log, *args, matrix=True)
        else : raise ValueError('"mode" should either be "sum" or "log".')

    else :
        if   mode == "sum" :
            genconv  = GenericKernelProduct().apply
            formula  = "("+kernel.formula_sum + " * B)"
        elif mode == "log" :
            genconv  = GenericLogSumExp().apply
            formula  = "("+kernel.formula_log + " + B)"
        elif mode == "log_scaled" :
            genconv  = GenericKernelProduct().apply
            formula  = "( Exp("+kernel.formula_log + "+ A_LOG + B_LOG) * B)"
        elif mode == "log_scaled_log" :
            genconv  = GenericLogSumExp().apply
            formula  = "("+kernel.formula_log + "+ A_LOG + B_LOG + B)"
        elif mode == "log_primal" :
            genconv  = GenericKernelProduct().apply
            formula  = "( (A_LOG + B_LOG - IntCst(1)) * Exp("+kernel.formula_log + "+ A_LOG + B_LOG) )"
        elif mode == "log_cost" :
            genconv  = GenericKernelProduct().apply
            formula  = "( (-"+kernel.formula_log+") * Exp("+kernel.formula_log + "+ A_LOG + B_LOG) )"
        else : raise ValueError('"mode" should either be "sum" or "log".')
        
        if   kernel.features == "locations" :
            nvars    = 3
            dimpoint = args[1].size(1) ; dimout = args[3].size(1)
            aliases  = ["DIMPOINT = "+str(dimpoint), "DIMOUT = "+str(dimout),
                        "G = Pm(0)"          ,   # 1st parameter
                        "X = Vx(0,DIMPOINT)" ,   # 1st variable, dim DIM,    indexed by i
                        "Y = Vy(1,DIMPOINT)" ,   # 2nd variable, dim DIM,    indexed by j
                        "B = Vy(2,DIMOUT  )" ]   # 3rd variable, dim DIMOUT, indexed by j
            # stands for:     R_i   ,   G  ,      X_i    ,      Y_j    ,     B_j    .
            signature = [ (dimout,0), (1,2), (dimpoint,0), (dimpoint,1), (dimout,1) ]

        elif kernel.features == "locations+directions" :
            nvars    = 5
            dimpoint = args[1].size(1) ; dimout = args[6].size(1)
        
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

        elif kernel.features == "locations+directions+values" :
            nvars    = 7
            dimpoint = args[1].size(1) ; dimout = args[9].size(1) ; dimsignal = args[7].size(1)
            aliases  = ["DIMPOINT = "+str(dimpoint), "DIMOUT = "+str(dimout), "DIMSIGNAL = "+str(dimsignal),
                        "G = Pm(0)"           ,   # 1st parameter
                        "H = Pm(1)"           ,   # 2nd parameter
                        "I = Pm(2)"           ,   # 3rd parameter
                        "X = Vx(0,DIMPOINT) " ,   # 1st variable, dim DIMPOINT,    indexed by i
                        "Y = Vy(1,DIMPOINT) " ,   # 2nd variable, dim DIMPOINT,    indexed by j
                        "U = Vx(2,DIMPOINT) " ,   # 3rd variable, dim DIMPOINT,    indexed by i
                        "V = Vy(3,DIMPOINT) " ,   # 4th variable, dim DIMPOINT,    indexed by j
                        "S = Vx(4,DIMSIGNAL)" ,   # 5th variable, dim DIMSIGNAL,   indexed by i
                        "T = Vy(5,DIMSIGNAL)" ,   # 6th variable, dim DIMSIGNAL,   indexed by j
                        "B = Vy(6,DIMOUT  ) " ]   # 7th variable, dim DIMOUT,      indexed by j
            # stands for:     R_i   ,   G  ,       X_i    ,       Y_j    ,
            signature = [ (dimout,0), (1,2),  (dimpoint,0),  (dimpoint,1), \
            #                           H  ,       U_i    ,       V_j    ,
                                      (1,2),  (dimpoint,0),  (dimpoint,1), \
            #                           I  ,       S_i    ,       T_j    ,
                                      (1,2), (dimsignal,0), (dimsignal,1), \
            #                              B_j    .
                                      (dimout,1) ]

        if mode in ("log_scaled", "log_scaled_log", "log_primal", "log_cost") :
            aliases += ["A_LOG = Vx("+str(nvars  )+",1)",
                        "B_LOG = Vy("+str(nvars+1)+",1)"]
            #               A_LOG_i , B_LOG_j
            signature += [   (1,0)  , (1,1)   ]
        sum_index = 0 # the output vector is indexed by "i" (CAT=0)
        return genconv( backend, aliases, formula, signature, sum_index, *args )
        
