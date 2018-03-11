from .generic_sum       import GenericSum
from .generic_logsumexp import GenericLogSumExp

from .utils import _scalar_products, _weighted_squared_distances, _log_sum_exp

def extract_metric_parameters(G) :
    """
    From the shape of the Variable G, infers if it is supposed
    to be used as a fixed parameter or as a "j" variable,
    and whether it represents a scalar, a diagonal matrix
    or a full symmetric matrix.
    """
    if len(G.shape) == 1 :
        G_cat = 2
        G_dim = G.shape[0]
    elif len(G.shape) == 2 :
        G_cat = 1
        G_dim = G.shape[1]
    else :
        raise ValueError("A 'metric' parameter is expected to be of dimension 1 or 2.")

    G_str = ["Vx", "Vy", "Pm"][G_cat]
    return G_dim, G_cat, G_str


def apply_routine(features, routine, *args):
    """PyTorch bindings."""
    if features == "locations":
        g, x, y, b = args
        K = routine(g=g, x=x, y=y, gxmy2=_weighted_squared_distances(g, x, y))

    elif features == "locations+directions":
        g, x, y, h, u, v, b = args
        K = routine(g=g, x=x, y=y, gxmy2=_weighted_squared_distances(g, x, y), \
                    h=h, u=u, v=v, usv=_scalar_products(u, v))

    elif features == "locations+directions+values":
        g, x, y, h, u, v, i, s, t, b = args
        K = routine(g=g, x=x, y=y, gxmy2=_weighted_squared_distances(g, x, y), \
                    h=h, u=u, v=v,  usv=_scalar_products(u, v), \
                    i=i, s=s, t=t, ismt2=_weighted_squared_distances(i, s, t))
    else:
        raise ValueError("This number of arguments is not supported!")
    return K


def _features_kernel(features, routine, *args, matrix=False):
    K = apply_routine(features, routine, *args)
    b = args[-1]
    return K if matrix else K @ b


def _features_kernel_log(features, routine, *args, matrix=False):
    K_log = apply_routine(features, routine, *args)
    b_log = args[-1]
    return K_log if matrix else _log_sum_exp(K_log + b_log.view(1, -1), 1).view(-1, 1)


def _features_kernel_log_scaled(features, routine, *args, matrix=False):
    a_log, b_log = args[-2:]  # scaling coefficients, typically given as output of the Sinkhorn loop
    K_log = _features_kernel_log(features, routine, *args[:-2], matrix=True)
    aKb_log = (a_log.view(-1, 1) + b_log.view(1, -1)) + K_log
    return aKb_log if matrix else aKb_log.exp() @ args[-3]


def _features_kernel_log_scaled_log(features, routine, *args, matrix=False):
    a_log, b_log = args[-2:]  # scaling coefficients, typically given as output of the Sinkhorn loop
    K_log = _features_kernel_log(features, routine, *args[:-2], matrix=True)
    aKb_log = (a_log.view(-1, 1) + b_log.view(1, -1)) + K_log
    return aKb_log if matrix else _log_sum_exp(aKb_log + args[-3].view(1, -1), 1).view(-1, 1)


def _features_kernel_log_primal(features, routine, *args, matrix=False):
    a_log, b_log, u, v = args[-4:]  # scaling coefficients, typically given as output of the Sinkhorn loop
    K_log = _features_kernel_log(features, routine, *args[:-4], matrix=True)
    aKb_log = (a_log.view(-1, 1) + b_log.view(1, -1)) + K_log
    minus_CGamma = (u.view(-1, 1) + v.view(1, -1) - 1.) * aKb_log.exp()
    return minus_CGamma if matrix else minus_CGamma.sum(1).view(-1, 1)


def _features_kernel_log_cost(features, routine, *args, matrix=False):
    a_log, b_log = args[-2:]  # scaling coefficients, typically given as output of the Sinkhorn loop
    K_log = _features_kernel_log(features, routine, *args[:-2], matrix=True)
    aKb_log = (a_log.view(-1, 1) + b_log.view(1, -1)) + K_log
    minus_CGamma = - K_log * aKb_log.exp()
    return minus_CGamma if matrix else minus_CGamma.sum(1).view(-1, 1)


def _features_kernel_log_barycenter(features, routine, *args, matrix=False):
    a_log, b_log, b2 = args[-3:]  # scaling coefficients, typically given as output of the Sinkhorn loop
    K_log = _features_kernel_log(features, routine, *args[:-3], matrix=True)
    aKb_log = (a_log.view(-1, 1) + b_log.view(1, -1)) + K_log
    aKb = aKb_log.exp()
    return aKb_log if matrix else ((aKb @ args[-4]) - (aKb.sum(1).view(-1, 1) * b2)).contiguous()


def FeaturesKP(kernel, *args, mode="sum", backend="auto", bonus_args=None):
    """
    *args = g,x,y, h,u,v, i,s,t, b
    """
    if bonus_args is not None:     args += tuple(bonus_args)

    if backend == "pytorch":
        if mode == "sum":
            return _features_kernel(kernel.features, kernel.routine_sum, *args)
        elif mode == "log":
            return _features_kernel_log(kernel.features, kernel.routine_log, *args)
        elif mode == "log_scaled":
            return _features_kernel_log_scaled(kernel.features, kernel.routine_log, *args)
        elif mode == "log_scaled_log":
            return _features_kernel_log_scaled_log(kernel.features, kernel.routine_log, *args)
        elif mode == "log_primal":
            return _features_kernel_log_primal(kernel.features, kernel.routine_log, *args)
        elif mode == "log_cost":
            return _features_kernel_log_cost(kernel.features, kernel.routine_log, *args)
        elif mode == "log_barycenter":
            return _features_kernel_log_barycenter(kernel.features, kernel.routine_log, *args)
        else:
            raise ValueError('"mode" should either be "sum" or "log".')
    elif backend == "matrix":
        if mode == "sum":
            return _features_kernel(kernel.features, kernel.routine_sum, *args, matrix=True)
        elif mode == "log":
            return _features_kernel_log(kernel.features, kernel.routine_log, *args, matrix=True)
        elif mode == "log_scaled":
            return _features_kernel_log_scaled(kernel.features, kernel.routine_log, *args, matrix=True)
        elif mode == "log_scaled_log":
            return _features_kernel_log_scaled_log(kernel.features, kernel.routine_log, *args, matrix=True)
        elif mode == "log_primal":
            return _features_kernel_log_primal(kernel.features, kernel.routine_log, *args, matrix=True)
        elif mode == "log_cost":
            return _features_kernel_log_cost(kernel.features, kernel.routine_log, *args, matrix=True)
        elif mode == "log_barycenter":
            return _features_kernel_log_barycenter(kernel.features, kernel.routine_log, *args, matrix=True)
        else:
            raise ValueError('"mode" should either be "sum" or "log".')

    else:
        if mode == "sum":
            genconv = GenericSum().apply
            formula = "(" + kernel.formula_sum + " * B)"
        elif mode == "log":
            genconv = GenericLogSumExp().apply
            formula = "(" + kernel.formula_log + " + B)"
        elif mode == "log_scaled":
            genconv = GenericSum().apply
            formula = "( Exp(" + kernel.formula_log + "+ A_LOG + B_LOG) * B)"
        elif mode == "log_scaled_log":
            genconv = GenericLogSumExp().apply
            formula = "(" + kernel.formula_log + "+ A_LOG + B_LOG + B)"
        elif mode == "log_primal":
            genconv = GenericSum().apply
            formula = "( (A_LOG2 + B_LOG2 - IntCst(1)) * Exp(" + kernel.formula_log + "+ A_LOG + B_LOG) )"
        elif mode == "log_cost":
            genconv = GenericSum().apply
            formula = "( (-" + kernel.formula_log + ") * Exp(" + kernel.formula_log + "+ A_LOG + B_LOG) )"
        elif mode == "log_barycenter":
            genconv = GenericSum().apply
            formula = "( Exp(" + kernel.formula_log + "+ A_LOG + B_LOG) * (B-B2))"
        else:
            raise ValueError('"mode" should either be "sum" or "log".')

        if kernel.features == "locations":
            (G,X,Y,B) = args
            dimpoint = X.size(1)
            dimout   = B.size(1)
            G_dim, G_cat, G_str = extract_metric_parameters(G)
            nvars = 4

            aliases = ["G = "+G_str+"(0,"+str(G_dim)+") ",  # parameter/j-variable
                       "X = Vx(1," + str(dimpoint) + ") ",  # variable, dim DIM,    indexed by i
                       "Y = Vy(2," + str(dimpoint) + ") ",  # variable, dim DIM,    indexed by j
                       "B = Vy(3," + str(dimout)   + ") "]  # variable, dim DIMOUT, indexed by j
            # stands for:     R_i   ,        G      ,       X_i    ,      Y_j    ,     B_j    .
            signature = [(dimout, 0), (G_dim, G_cat), (dimpoint, 0), (dimpoint, 1), (dimout, 1)]

        elif kernel.features == "locations+directions":
            (G,X,Y,H,U,V,B) = args
            dimpoint = X.size(1)
            dimout   = B.size(1)
            G_dim, G_cat, G_str = extract_metric_parameters(G)
            H_dim, H_cat, H_str = extract_metric_parameters(H)
            nvars = 7

            aliases = ["G = "+G_str+"(0,"+str(G_dim)+") ",  # parameter/j-variable
                       "X = Vx(1," + str(dimpoint) + ") ",  # variable, dim DIM,    indexed by i
                       "Y = Vy(2," + str(dimpoint) + ") ",  # variable, dim DIM,    indexed by j
                       "H = "+H_str+"(3,"+str(H_dim)+") ",  # parameter/j-variable
                       "U = Vx(4," + str(dimpoint) + ") ",  # variable, dim DIM,    indexed by i
                       "V = Vy(5," + str(dimpoint) + ") ",  # variable, dim DIM,    indexed by j
                       "B = Vy(6," + str(dimout)   + ") "]  # variable, dim DIMOUT, indexed by j
            # stands for:     R_i   ,       G       ,       X_i    ,       Y_j    ,
            signature = [(dimout, 0), (G_dim, G_cat), (dimpoint, 0), (dimpoint, 1), \
                         #                  H       ,       U_i    ,       V_j    ,
                                      (H_dim, H_cat), (dimpoint, 0), (dimpoint, 1), \
                         #                 B_j      .
                                    (dimout, 1)]

        elif kernel.features == "locations+directions+values":
            (G,X,Y,H,U,V,I,S,T,B) = args
            dimpoint  = X.size(1)
            dimsignal = S.size(1)
            dimout    = B.size(1)
            G_dim, G_cat, G_str = extract_metric_parameters(G)
            H_dim, H_cat, H_str = extract_metric_parameters(H)
            I_dim, I_cat, I_str = extract_metric_parameters(I)
            nvars = 10

            aliases = ["G = "+G_str+"(0,"+str(G_dim)+") ",  # parameter/j-variable
                       "X = Vx(1," + str(dimpoint) + ") ",  # variable, dim DIMPOINT,    indexed by i
                       "Y = Vy(2," + str(dimpoint) + ") ",  # variable, dim DIMPOINT,    indexed by j
                       "H = "+H_str+"(3,"+str(H_dim)+") ",  # parameter/j-variable
                       "U = Vx(4," + str(dimpoint) + ") ",  # variable, dim DIMPOINT,    indexed by i
                       "V = Vy(5," + str(dimpoint) + ") ",  # variable, dim DIMPOINT,    indexed by j
                       "I = "+I_str+"(6,"+str(I_dim)+") ",  # parameter/j-variable
                       "S = Vx(7," + str(dimsignal)+ ") ",  # variable, dim DIMSIGNAL,   indexed by i
                       "T = Vy(8," + str(dimsignal)+ ")",   # variable, dim DIMSIGNAL,   indexed by j
                       "B = Vy(9," + str(dimout)   + ") "]  # variable, dim DIMOUT,      indexed by j
            # stands for:     R_i   ,       G       ,       X_i    ,       Y_j    ,
            signature = [(dimout, 0), (G_dim, G_cat), (dimpoint, 0), (dimpoint, 1), \
                         #                  H       ,       U_i    ,       V_j    ,
                                      (H_dim, H_cat), (dimpoint, 0), (dimpoint, 1), \
                         #                  I       ,       S_i    ,       T_j    ,
                                      (I_dim, I_cat), (dimsignal, 0), (dimsignal, 1), \
                         #                  B_j    .
                                      (dimout, 1)]

        if mode in ("log_scaled", "log_scaled_log", "log_primal", "log_cost", "log_barycenter"):
            aliases += ["A_LOG = Vx(" + str(nvars) + ",1)",
                        "B_LOG = Vy(" + str(nvars + 1) + ",1)"]
            #               A_LOG_i , B_LOG_j
            signature += [(1, 0), (1, 1)]
        if mode == "log_primal":
            aliases += ["A_LOG2 = Vx(" + str(nvars + 2) + ",1)",
                        "B_LOG2 = Vy(" + str(nvars + 3) + ",1)"]
            #               A_LOG2_i , B_LOG2_j
            signature += [(1, 0), (1, 1)]
        if mode == "log_barycenter":
            aliases += ["B2 = Vx(" + str(nvars + 2) + "," + str(dimout) + ")"]
            #                  B2_i
            signature += [(dimout, 0)]

        sum_index = 0  # the output vector is indexed by "i" (CAT=0)
        return genconv(backend, aliases, formula, signature, sum_index, *args)
