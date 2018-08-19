import numpy as np

from pykeops.common.utils import axis2cat


def squared_distances(x, y):
    return np.sum((x[:,np.newaxis,:] - y[np.newaxis,:,:]) ** 2, axis=2)


def differences(x, y):
    return (x.T[:,:,np.newaxis] - y.T[:,np.newaxis,:])


def np_kernel_sphere(nalpha, nbeta, s, kernel) :
    prs = nalpha @ nbeta.T
    if   kernel == "binet"               : return prs**2
    elif kernel == "linear"              : return prs
    elif kernel == "gaussian_unoriented" : return np.exp( (-2.0 + 2.0 * prs*prs) / (s*s))
    elif kernel == "gaussian_oriented"   : return np.exp( (-2.0 + 2.0 * prs) / (s*s))


def np_kernel(x, y, s, kernel) :
    sq = squared_distances(x, y)
    if   kernel == "gaussian"  : return np.exp( -sq / (s*s))
    elif kernel == "laplacian" : return np.exp( -np.sqrt(sq) / s)
    elif kernel == "cauchy"    : return  1. / ( 1 + sq / (s*s) )
    elif kernel == "inverse_multiquadric" : return np.sqrt(  1. / ( 1 + sq/(s*s) ) )


def log_np_kernel(x, y, s, kernel) :
    sq = squared_distances(x, y)
    if   kernel == "gaussian"  : return -sq / (s*s)
    elif kernel == "laplacian" : return -np.sqrt(sq) / s
    elif kernel == "cauchy"    : return -np.log( 1 + sq / (s*s) )
    elif kernel == "inverse_multiquadric" : return -.5*np.log(1. + sq / (s*s) )


def grad_np_kernel(x, y, s, kernel) :
    sq = squared_distances(x, y)
    if   kernel == "gaussian"  : return - np.exp(-sq / (s*s)) / (s*s)
    elif kernel == "laplacian" : t = -np.sqrt(sq / (s*s)) ; return  np.exp(t) / (2*s*s*t)
    elif kernel == "cauchy"    : return -1. / (s * (sq/(s*s) + 1) )**2 
    elif kernel == "inverse_multiquadric"    : return -.5 / ((s**2) * ( (sq/(s*s) + 1)**1.5) )


def chain_rules(q,ax,by,Aa,p):
    res = np.zeros(ax.shape).astype('float32')
    for i in range(ax.shape[1]):
        #Computation of 2*|x_i -x_j|*exp(-|x_i -x_j|^2/(lam^2))/(lam^2)
        ximyj = (np.tile(ax[:,i],[by.shape[0],1]).T - np.tile(by[:,i],[ax.shape[0],1])) 
        res[:,i] = np.sum(q * ((2 * ximyj * Aa) @ p),axis=1)
    return res


def log_sum_exp(mat, axis=0):
    """
    Computes the log-sum-exp of a matrix with a numerically stable scheme,
    in the user-defined summation dimension: exp is never applied
    to a number >= 0, and in each summation row, there is at least
    one "exp(0)" to stabilize the sum.

    For instance, if dim = 1 and mat is a 2d array, we output
                log( sum_j exp( mat[i,j] ))
    by factoring out the row-wise maximas.
    """
    max_rc = mat.max(axis=axis)
    return max_rc + np.log(np.sum(np.exp(mat -  np.expand_dims(max_rc, axis=axis)), axis=axis))


def assert_contiguous(x):
    """Non-contiguous arrays are a mess to work with,
    so we require contiguous arrays from the user."""
    if not x.flags.c_contiguous: raise ValueError("Please provide 'C-contiguous' numpy arrays.")


def ndims(x):
    return x.ndim


def dtype(x):
    return x.dtype


def size(x):
    return x.size


def to_ctype_pointer(x):
    from ctypes import POINTER, c_float
    assert_contiguous(x)
    return x.ctypes.data_as(POINTER(c_float))


def vect_from_list(l):
    return np.hstack(l)


def is_on_device(x):
    return False
