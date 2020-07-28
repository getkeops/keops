import torch
from pykeops.common.utils import get_tools
from math import sqrt
import warnings


#############################################
# CG_revcom with Python dictionnary
#############################################

def cg(linop, b, binding, x=None, eps=None, maxiter=None, callback=None, check_cond=False):
    if binding not in ("torch", "numpy", "pytorch"):
        raise ValueError(
            "Language not supported, please use numpy, torch or pytorch.")

    tools = get_tools(binding)

    # we don't need cuda with numpy (at least i think so)
    is_cuda = True if (binding == 'torch' or binding ==
                       'pytorch') and torch.cuda.is_available() else False
    device = torch.device("cuda") if is_cuda else torch.device('cpu')

    b, x, replaced = check_dims(b, x, tools, is_cuda)
    n, m = b.shape

    if eps == None:
        eps = 1e-6 * sqrt((b ** 2).sum())

    if maxiter == None:
        maxiter = 10 * n

    if check_cond:
        from pykeops.common.power_iteration import bootleg_inv_power_cond_big as cond_big
        cond_too_big = cond_big(linop, n, binding, device)
        if cond_too_big:
            warnings.warn(
                "Warning ----------- Condition number might be too large.")

    # define the functions needed along the iterations
    if binding == "numpy":
        p, q, r = tools.zeros((n, m), dtype=b.dtype), tools.zeros(
            (n, m), dtype=b.dtype), tools.zeros((n, m), dtype=b.dtype)
        scal1, scal2 = tools.zeros(1, dtype=b.dtype), tools.zeros(
            1, dtype=b.dtype)  # init the scala values

    else:
        p, q, r = tools.zeros((n, m), dtype=b.dtype, device=device), tools.zeros(
            (n, m), dtype=b.dtype, device=device), tools.zeros((n, m), dtype=b.dtype, device=device)
        scal1, scal2 = tools.zeros(1, dtype=b.dtype, device=device), tools.zeros(
            1, dtype=b.dtype, device=device)  # init the scala values

    def init_iter(linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_):  # revc -> cg
        r = tools.copy(b) if replaced else (b - linop(x))
        scal1 = (r ** 2).sum()
        job_cg = "check"
        return job_cg, x, r, p, q, scal1, scal2, iter_

    def check_resid(linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_):  # cg -> revc
        if scal1 <= eps**2 or scal1 != scal1:
            job_rev = "stop"
        else:
            iter_ += 1
            job_rev = "direction_next" if iter_ > 1 else "direction_first"
        return job_rev, x, r, p, q, scal1, scal2, iter_

    def first_direct(linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_):  # revc -> cg
        p = tools.copy(r)
        job_cg = "matvec_p"
        return job_cg, x, r, p, q, scal1, scal2, iter_

    def matvec_p(linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_):  # cg -> revc
        q = linop(p)
        job_rev = "update"
        return job_rev, x, r, p, q, scal1, scal2, iter_

    def update(linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_):  # revc -> cg
        alpha = scal1 / (p * q).sum()
        x += alpha * p
        r -= alpha * q
        scal2 = scal1
        job_cg = "check"
        return job_cg, x, r, p, q, scal1, scal2, iter_

    def next_direct(linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_):  # revc -> cg
        scal1 = (r ** 2).sum()
        p = r + (scal1 / scal2) * p
        job_cg = "matvec_p"
        return job_cg, x, r, p, q, scal1, scal2, iter_

    jobs_cg = {"matvec_p": matvec_p,
               "check": check_resid
               }

    jobs_revcom = {
        "init": init_iter,
        "update": update,
        "direction_first": first_direct,
        "direction_next": next_direct
    }

    iter_ = 0
    job_rev = "init"
    job_cg = None

    while iter_ <=maxiter:
        if job_cg == "check" and callback is not None:
            if iter_ > 1:
                callback(x)
        job_cg, x, r, p, q, scal1, scal2, iter_ = jobs_revcom[job_rev](
            linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_)
        job_rev, x, r, p, q, scal1, scal2, iter_ = jobs_cg[job_cg](
            linop, x, r, p, q, b, scal1, scal2, eps, replaced, iter_)

        if job_rev == "stop":
            break
    if (iter_ - 1) == maxiter:
        warnings.warn("Warning ----------- Conjugate gradient reached maximum iteration !")

    return x, iter_


####################################
# Sanity checks
####################################

def check_dims(b, x, tools, cuda_avlb):  # x is always of b's shape
    try:
        nrow, ncol = b.shape
    except ValueError:
        b = b.reshape(-1, 1)
        nrow, ncol = b.shape

    x_replaced = False

    if x is None:  # check x shape and initiate it if needed
        x = tools.zeros((nrow, ncol), dtype=b.dtype, device=torch.device('cuda')) if cuda_avlb \
            else tools.zeros((nrow, ncol), dtype=b.dtype)
        x_replaced = True
    elif (nrow, ncol) != x.shape:  # add sth to check if x is on the same device as b if torch is used!
        if x.shape == (nrow,):
            x = x.reshape((nrow, ncol))
        else:
            raise ValueError("Mismatch between shapes of b {} and shape of x {}.".format(
                (nrow, nrow), x.shape))
        if x.dtype != b.dtype:
            raise ValueError("Type of given x {} is not compatible with type of b {}.".format(x.dtype, b.dtype))

    return b, x, x_replaced
