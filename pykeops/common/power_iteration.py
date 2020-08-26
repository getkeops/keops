import torch
import numpy as np
import warnings
from math import sqrt

from pykeops.common.utils import get_tools
from pykeops.common.cg import cg

#####################
# Power iteration
#####################


def random_draw_np(size, device, dtype='float32'):
    return np.random.rand(size, 1).astype(dtype)


def random_draw_torch(size,  device, dtype=torch.float32):
    return torch.rand(size, 1, device=device, dtype=dtype)


def power_it_ray(linop, size, binding, device, eps=1e-6):
    r""" Compute the eigenvalue of maximum magnitude for a linear operator.

    Args:
        linop: a linear operator that, when called, computes the matrix-vector product.
            size (int): dimension of the linear operator.
        binding (string): torch, pytorch or numpy.
        device (torch.device): use GPU or CPU, ``torch.device('cuda')`` or ``torch.device("cpu")``*
            for example.

    Keyword Args:
        eps (float, default=1e-6): precision for the acceptable distance between two iterates of
            the eigenvalue.
    
    Returns:
        lambd_ (float): the eigenvalue of maximum magnitude for ``linop``.
            A warning is displayed if the algorithm didn't converge.
    """
    random = random_draw_np if binding == "numpy" else random_draw_torch
    x = random(size, device)
    x = x / sqrt((x ** 2).sum())
    maxiter = 10 * size
    k = 0
    while k <= maxiter:
        y = linop(x)
        norm_y = sqrt((y ** 2).sum())
        z = y / norm_y
        lambd_ = (z.T @ linop(z))
        if k > 0 and (old_lambd - lambd_) ** 2 <= eps ** 2:
            break
        old_lambd = lambd_
        x = z
        k += 1
    if (k - 1) == maxiter:
        warnings.warn(
            "Warning ----------- Power iteration method did not converge !")
    return lambd_


def bootleg_inv_power_cond_big(linop, size, binding, device, maxcond=500, maxiter=50):
    lambda_max = power_it_ray(linop, size, binding, device)
    thresh = lambda_max / maxcond
    k = 0
    vp = [maxcond]
    random = random_draw_np if binding == "numpy" else random_draw_torch
    x = random(size, device)
    while k <= maxiter:
        x = cg(linop, x, binding)[0]
        x = x / sqrt((x ** 2).sum())
        vp.append(x.T @ linop(x))
        if vp[k] <= thresh:
            cond_too_big = True
            break
        if k >=1 and (vp[k]-vp[k-1]) ** 2 <= 1e-10: #cv
            k = maxiter #exit
        k += 1
    if (k - 1) == maxiter:
        cond_too_big = False
    return cond_too_big
