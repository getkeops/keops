from pykeops.torch import Genred
from pykeops.torch.kernel_product.formula import extract_metric_parameters, _scalar_products, _log_sum_exp, _weighted_squared_distances


def apply_routine(routine, gs, xs, ys):
    """
    PyTorch bindings.
    """
    gxmy2s, xsys = [], []
    try :
        for (i, (g, (v_ind, x_, y_, gxmy2_, xsy_))) in enumerate(zip(gs, routine.vars_needed)):
            gxmy2s.append(_weighted_squared_distances(g, xs[v_ind], ys[v_ind]) if gxmy2_ else None)
            xsys.append(_scalar_products(xs[v_ind], ys[v_ind]) if xsy_ else None)
    except AttributeError :
        # The user has provided a handmade routine:
        gxmy2s, xsys = None

    return routine(x=xs, y=ys, g=gs, gxmy2=gxmy2s, xsy=xsys)


def _features_kernel(routine, gs, xs, ys, bs, matrix=False):
    K = apply_routine(routine, gs, xs, ys)
    (b,) = bs
    return K if matrix else K @ b


def _features_kernel_log(routine, gs, xs, ys, bs, matrix=False):
    K_log = apply_routine(routine, gs, xs, ys)
    (b_log,) = bs
    return K_log if matrix else _log_sum_exp(K_log + b_log.view(1, -1), 1).view(-1, 1)


# BONUS MODES (not documented in this version) ===================================================
def _features_kernel_log_scaled(routine, gs, xs, ys, bs, matrix=False):
    b, A_log, B_log = bs  # scaling coefficients, typically given as output of the Sinkhorn loop
    K_log = _features_kernel_log(routine, gs, xs, ys, (b,), matrix=True)
    aKb_log = (A_log.view(-1, 1) + B_log.view(1, -1)) + K_log
    return aKb_log if matrix else aKb_log.exp() @ b


def _features_kernel_log_scaled_lse(routine, gs, xs, ys, bs, matrix=False):
    b_log, A_log, B_log = bs  # scaling coefficients, typically given as output of the Sinkhorn loop
    K_log = _features_kernel_log(routine, gs, xs, ys, (b_log,), matrix=True)
    aKb_log = (A_log.view(-1, 1) + B_log.view(1, -1)) + K_log
    return aKb_log if matrix else _log_sum_exp(aKb_log + b_log.view(1, -1), 1).view(-1, 1)


def _features_kernel_log_scaled_barycenter(routine, gs, xs, ys, bs, matrix=False):
    b, A_log, B_log, b2 = bs  # scaling coefficients, typically given as output of the Sinkhorn loop
    K_log = _features_kernel_log(routine, gs, xs, ys, (b,), matrix=True)
    aKb_log = (A_log.view(-1, 1) + B_log.view(1, -1)) + K_log
    aKb = aKb_log.exp()
    return aKb_log if matrix else ((aKb @ b) - (aKb.sum(1).view(-1, 1) * b2)).contiguous()


def _features_kernel_lse_mult_i(routine, gs, xs, ys, bs, matrix=False):
    b_log, a_i   = bs  # scaling coefficient
    K_log = _features_kernel_log(routine, gs, xs, ys, (b_log,), matrix=True)
    K_log = a_i.view(-1,1) * K_log
    return K_log if matrix else _log_sum_exp(K_log + b_log.view(1, -1), 1).view(-1, 1)


def _features_kernel_sinkhorn_primal(routine, gs, xs, ys, bs, matrix=False):
    A_log, B_log, u, v = bs  # scaling coefficients, typically given as output of the Sinkhorn loop
    K_log = _features_kernel_log(routine, gs, xs, ys, (None,), matrix=True)
    aKb_log = (A_log.view(-1, 1) + B_log.view(1, -1)) + K_log
    minus_CGamma = (u.view(-1, 1) + v.view(1, -1) - 1.) * aKb_log.exp()
    return minus_CGamma if matrix else minus_CGamma.sum(1).view(-1, 1)


def _features_kernel_sinkhorn_cost(routine, gs, xs, ys, bs, matrix=False):
    A_log, B_log = bs # scaling coefficients, typically given as output of the Sinkhorn loop
    K_log = _features_kernel_log(routine, gs, xs, ys, (None,), matrix=True)
    aKb_log = (A_log.view(-1, 1) + B_log.view(1, -1)) + K_log
    minus_CGamma = - K_log * aKb_log.exp()
    return minus_CGamma if matrix else minus_CGamma.sum(1).view(-1, 1)


# ==============================================================================================
pytorch_routines = {
    'sum'                  : ('sum', _features_kernel),
    'lse'                  : ('log', _features_kernel_log),
    'log_scaled'           : ('log', _features_kernel_log_scaled),
    'log_scaled_lse'       : ('log', _features_kernel_log_scaled_lse),
    'log_scaled_barycenter': ('log', _features_kernel_log_scaled_barycenter),
    'lse_mult_i'           : ('log', _features_kernel_lse_mult_i),
    'sinkhorn_primal'      : ('log', _features_kernel_sinkhorn_primal),
    'sinkhorn_cost'        : ('log', _features_kernel_sinkhorn_cost),
}


keops_routines = {
    'sum'                  : ('Sum'      , '({f_sum} * B_0)',                                                [1]),
    'lse'                  : ('LogSumExp', '({f_log} + B_0)',                                                [1]),
    'log_scaled'           : ('Sum'      , '(Exp({f_log} + B_1 + B_2) * B_0)',                         [1, 0, 1]),
    'log_scaled_lse'       : ('LogSumExp', '({f_log} + B_1 + B_2 + B_0)',                              [1, 0, 1]),
    'log_scaled_barycenter': ('Sum'      , '(Exp({f_log} + B_1 + B_2) * (B_0-B_3))',               [1, 0, 1, 0]),
    'lse_mult_i'           : ('LogSumExp', '((B_1 * {f_log}) + B_0)',                                     [1, 0]),
    'sinkhorn_primal'      : ('Sum'      , '((B_2 + B_3 - IntCst(1)) * Exp({f_log} + B_0 + B_1))', [0, 1, 0, 1]),
    'sinkhorn_cost'        : ('Sum'      , '((- {f_log}) * Exp({f_log} + B_0 + B_1))',                    [0, 1]),
}


def FeaturesKP(kernel, gs, xs, ys, bs, mode='sum', backend='auto', dtype='float32'):
    if backend in ['pytorch', 'matrix']:
        domain, torch_map = pytorch_routines[mode]
        if domain == 'sum':
            routine = kernel.routine_sum
        elif domain == 'log':
            routine = kernel.routine_log

        return torch_map(routine, gs, xs, ys, bs, matrix=(backend == 'matrix'))

    else:
        red, formula, bs_cat = keops_routines[mode]

        formula = formula.format(f_sum=kernel.formula_sum, f_log=kernel.formula_log)

        # Given the output sizes, we must generate the appropriate list of aliases

        # We will store the arguments as follow :
        # [ G_0, G_1, ..., X_0, X_1, Y_0, Y_1, ...]
        full_args, aliases, index = [], [], 0  # tensor list, string list, current input arg

        # First, the G_i's
        for (i, g) in enumerate(gs):
            if g is not None:
                g_var, g_dim, g_cat, g_str = extract_metric_parameters(g) # example : Tensor(...), 3, 0, 'Vi'
                aliases.append('G_{g_ind} = {g_str}({index}, {g_dim})'.format(
                                g_ind=i, g_str=g_str, index=index, g_dim=g_dim))
                full_args.append(g_var)
                index += 1

        # Then, the X_i's
        for (i, x) in enumerate(xs):
            x_dim = x.size(1)
            aliases.append('X_{x_ind} = Vi({index}, {x_dim})'.format(
                             x_ind=i, index=index, x_dim=x_dim))
            full_args.append(x)
            index += 1

        # Then, the Y_j's
        for (j, y) in enumerate(ys):
            y_dim = y.size(1)
            aliases.append('Y_{y_ind} = Vj({index}, {y_dim})'.format(
                             y_ind=j, index=index, y_dim=y_dim))
            full_args.append(y)
            index += 1

        if not len(xs) == len(ys):
            raise ValueError("Kernel_product works with pairs of variables. The 'x'-list of features should thus have the same length as the 'y' one.")

        # Then, the B_i/j's
        for (i, (b, b_cat)) in enumerate(zip(bs, bs_cat)):
            b_dim = b.size(1)
            b_str = ['Vi', 'Vj', 'Pm'][b_cat]
            aliases.append('B_{b_ind} = {b_str}({index}, {b_dim})'.format(
                            b_ind=i, b_str=b_str, index=index, b_dim=b_dim))
            full_args.append(b)
            index += 1

        axis = 1  # the output vector is indexed by 'i' (CAT=0)
        genconv = Genred(formula, aliases, reduction_op=red, axis=axis, dtype=dtype)

        return genconv(*full_args, backend=backend)
