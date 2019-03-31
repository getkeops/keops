import numpy as np

from pykeops.numpy import Genred, default_dtype


class numpytools:
    norm = np.linalg.norm
    arraysum = np.sum
    exp = np.exp
    log = np.log
    
    @staticmethod
    def copy(x): return np.copy(x)
    
    @staticmethod
    def transpose(x): return x.T
    
    @staticmethod
    def numpy(x): return x
    
    @staticmethod
    def tile(*args): return np.tile(*args)
    
    @staticmethod
    def solve(*args): return np.linalg.solve(*args)
    
    @staticmethod
    def size(x): return x.size
    
    @staticmethod
    def view(x, s): return np.reshape(x, s)
    
    @staticmethod
    def long(x): return x.astype('int64')
    
    @staticmethod
    def dtype(x): return x.dtype.name
    
    @staticmethod
    def rand(m, n, dtype=default_dtype): return np.random.rand(m, n).astype(dtype)
    
    @staticmethod
    def randn(m, n, dtype=default_dtype): return np.random.randn(m, n).astype(dtype)
    
    @staticmethod
    def zeros(shape, dtype=default_dtype): return np.zeros(shape).astype(dtype)
    
    @staticmethod
    def eye(n, dtype=default_dtype): return np.eye(n).astype(dtype)
    
    @staticmethod
    def array(x, dtype=default_dtype): return np.array(x).astype(dtype)


def squared_distances(x, y):
    x_norm = (x ** 2).sum(1).reshape(-1, 1)
    y_norm = (y ** 2).sum(1).reshape(1, -1)
    dist = x_norm + y_norm - 2.0 * np.matmul(x, y.T)
    return dist


def differences(x, y):
    return (x.T[:, :, np.newaxis] - y.T[:, np.newaxis, :])


def np_kernel_sphere(nalpha, nbeta, s, kernel):
    prs = nalpha @ nbeta.T
    if kernel == "binet":
        return prs ** 2
    elif kernel == "linear":
        return prs
    elif kernel == "gaussian_unoriented":
        return np.exp((-2.0 + 2.0 * prs * prs) / (s * s))
    elif kernel == "gaussian_oriented":
        return np.exp((-2.0 + 2.0 * prs) / (s * s))


def np_kernel(x, y, s, kernel):
    sq = squared_distances(x, y)
    if kernel == "gaussian":
        return np.exp(-sq / (s * s))
    elif kernel == "laplacian":
        return np.exp(-np.sqrt(sq) / s)
    elif kernel == "cauchy":
        return 1. / (1 + sq / (s * s))
    elif kernel == "inverse_multiquadric":
        return np.sqrt(1. / (1 + sq / (s * s)))


def log_np_kernel(x, y, s, kernel):
    sq = squared_distances(x, y)
    if kernel == "gaussian":
        return -sq / (s * s)
    elif kernel == "laplacian":
        return -np.sqrt(sq) / s
    elif kernel == "cauchy":
        return -np.log(1 + sq / (s * s))
    elif kernel == "inverse_multiquadric":
        return -.5 * np.log(1. + sq / (s * s))


def grad_np_kernel(x, y, s, kernel):
    sq = squared_distances(x, y)
    if kernel == "gaussian":
        return - np.exp(-sq / (s * s)) / (s * s)
    elif kernel == "laplacian":
        t = -np.sqrt(sq / (s * s)); return np.exp(t) / (2 * s * s * t)
    elif kernel == "cauchy":
        return -1. / (s * (sq / (s * s) + 1)) ** 2
    elif kernel == "inverse_multiquadric":
        return -.5 / ((s ** 2) * ((sq / (s * s) + 1) ** 1.5))


def chain_rules(q, ax, by, Aa, p):
    res = np.zeros(ax.shape).astype('float32')
    for i in range(ax.shape[1]):
        # Computation of 2*|x_i -x_j|*exp(-|x_i -x_j|^2/(lam^2))/(lam^2)
        ximyj = (np.tile(ax[:, i], [by.shape[0], 1]).T - np.tile(by[:, i], [ax.shape[0], 1]))
        res[:, i] = np.sum(q * ((2 * ximyj * Aa) @ p), axis=1)
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
    return max_rc + np.log(np.sum(np.exp(mat - np.expand_dims(max_rc, axis=axis)), axis=axis))


def IsGpuAvailable():
    # testing availability of Gpu: 
    try:
        import GPUtil
        useGpu = len(GPUtil.getGPUs()) > 0
    except:
        useGpu = False
    return useGpu


def WarmUpGpu():
    # dummy first calls for accurate timing in case of GPU use
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    aliases = ['x = Vi(1)',  # First arg   : i-variable, of size 1
               'y = Vj(1)',  # Second arg  : j-variable, of size 1
               'b = Vj(1)',  # Third arg  : j-variable, of size 1
               'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = Genred(formula, aliases, reduction_op='Sum', axis=1, dtype='float64')
    dum = np.random.rand(10, 1)
    dum2 = np.random.rand(10, 1)
    my_routine(dum, dum, dum2, np.array([1.0]))
    my_routine(dum, dum, dum2, np.array([1.0]))
