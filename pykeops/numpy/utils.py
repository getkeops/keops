import numpy as np
from pykeops.numpy import Genred
from pykeops.numpy.generic.generic_red import Genred_lowlevel

class numpytools :
    norm = np.linalg.norm
    arraysum = np.sum
    Genred = Genred
    Genred_lowlevel = Genred_lowlevel
    exp = np.exp
    log = np.log
    def __init__(self):
        self.copy = lambda x : np.copy(x)
        self.transpose = lambda x : x.T
        self.numpy = lambda x : x
        self.tile = lambda *args : np.tile(*args)
        self.solve = lambda *args : np.linalg.solve(*args)
        self.size = lambda x : x.size
        self.view = lambda x,s : np.reshape(x,s)
    def set_types(self,x):
        self.dtype = x.dtype.name
        self.rand = lambda m, n : np.random.rand(m,n,dtype=self.dtype)
        self.randn = lambda m, n : np.random.randn(m,n,dtype=self.dtype)
        self.zeros = lambda shape : np.zeros(shape,dtype=self.dtype)
        self.eye = lambda n : np.eye(n,dtype=self.dtype)
        self.array = lambda x : np.array(x,dtype=self.dtype)

def squared_distances(x, y):
    x_norm = (x ** 2).sum(1).reshape(-1, 1)
    y_norm = (y ** 2).sum(1).reshape(1, -1)
    dist = x_norm + y_norm - 2.0 * np.matmul(x, y.T)
    return dist

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
    return max_rc + np.log(np.sum(np.exp(mat - np.expand_dims(max_rc, axis=axis)), axis=axis))

def IsGpuAvailable():
    # testing availability of Gpu: 
    try:
        import GPUtil
        useGpu = len(GPUtil.getGPUs())>0
    except:
        useGpu = False
    return useGpu

def WarmUpGpu():
    # dummy first calls for accurate timing in case of GPU use
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    aliases = ['x = Vx(1)',  # First arg   : i-variable, of size 1
                 'y = Vy(1)',  # Second arg  : j-variable, of size 1
                 'b = Vy(1)',  # Third arg  : j-variable, of size 1
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = Genred(formula, aliases, reduction_op='Sum', axis=1, cuda_type='float64')
    dum = np.random.rand(10,1)
    dum2 = np.random.rand(10,1)
    my_routine(dum,dum,dum2,np.array([1.0]))
    my_routine(dum,dum,dum2,np.array([1.0]))


