import ctypes
from ctypes import POINTER, c_int, c_float
import os.path

# extract cuda_fshape_scp function pointer in the shared object cuda_fshape_scp_*.so
def get_cuda_fshape_scp_dx(name_geom , name_sig , name_var):
    """
    Loads the routine from the compiled .so file.
    """
    name = name_geom + name_sig + name_var
    dll_name = "cuda_fshape_scp_dx_" + name + ".so"

    dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep+ 'build' + os.path.sep + dll_name
    dll = ctypes.CDLL(dllabspath, mode=ctypes.RTLD_GLOBAL)
    func = dll.cudafshape_dx
    # Arguments :     1/sx^2,   1/sf^2, 1/st^2,     x,               y,                f,                  g,              alpha,            beta,             result,       dim-xy, dim-fg, dim-beta, nx,   ny
    func.argtypes = [c_float, c_float, c_float, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int,   c_int  , c_int, c_int]
    return func

# convenient python wrapper for __cuda_fshape_scp it does all job with types convertation from python ones to C++ ones 
def cuda_shape_scp_dx(x, y, f, g, alpha, beta, result, sigma_geom, sigma_sig, sigma_var = 1, kernel_geom = "gaussian", kernel_sig ="gaussian", kernel_var = "binet"):
    """
    Implements the operation :

    (x_i, y_j,f_i, g_j, alpha_i, beta_j)  -> partial_x( \sum_j kgeom(x_i,y_j) ksig(f_i,g_j) kvar(alpha_i,beta_j) )_i ,

    where kgeom, ksig, kvar are kernels function of parameter "sigmageom", "sigmasig", "sigmavar".

    NB : this function is useful to compute varifold like distance between shapes.
    """
    # From python to C float pointers and int :
    x_p      =  x.ctypes.data_as(POINTER(c_float))
    y_p      =  y.ctypes.data_as(POINTER(c_float))
    f_p      =  f.ctypes.data_as(POINTER(c_float))
    g_p      =  g.ctypes.data_as(POINTER(c_float))
    alpha_p  =  alpha.ctypes.data_as(POINTER(c_float))
    beta_p   =  beta.ctypes.data_as(POINTER(c_float))
    result_p =  result.ctypes.data_as(POINTER(c_float))

    nx = x.shape[0] ; ny = y.shape[0]

    dimPoint = x.shape[1]
    dimSig   = f.shape[1]
    dimVect  = beta.shape[1]

    ooSigma_geom2 = float(1/ (sigma_geom*sigma_geom)) # Compute this once and for all
    ooSigma_sig2  = float(1 / (sigma_sig*sigma_sig)) # Compute this once and for all
    ooSigma_var2  = float(1 / (sigma_var*sigma_var)) # Compute this once and for all

    # create __cuda_fshape_scp function with get_cuda_fshape_scp()
    __cuda_fshape_scp_dx = get_cuda_fshape_scp_dx(kernel_geom , kernel_sig , kernel_var)
    # Let's use our GPU, which works "in place" :
    __cuda_fshape_scp_dx(ooSigma_geom2,ooSigma_sig2,ooSigma_var2, x_p, y_p, f_p, g_p, alpha_p, beta_p, result_p, dimPoint,dimSig, dimVect, nx, ny )



if __name__ == '__main__':
    """
    testing the cuda kernel with a python  implementation
    """
    
    import numpy as np
    
   
    np.set_printoptions(linewidth=100)

    sizeX    = int(4)
    sizeY    = int(5)
    dimPoint = int(3)
    dimSig = int(1)
    dimVect  = int(3)
    sigma_geom  = 1.0
    sigma_sig  = 1.0
    sigma_var   = np.pi/2

    def np_kernel(x, y, s, kernel ="gaussian") :
        r2 = np.sum( (x[:,np.newaxis] - y[np.newaxis,]) **2, axis=2)
        if   kernel == "gaussian"  : return np.exp( -r2 / (s*s))
        elif kernel == "cauchy"    : return 1 /(1 + r2/(s*s))
        return -1

    def dnp_kernel(x, y, s, kernel ="gaussian") :
        r2 = np.sum( (x[:,np.newaxis] - y[np.newaxis,]) **2, axis=2)
        if   kernel == "gaussian"  : 
            return - np.exp( -r2 / (s*s)) / (s*s)
        elif kernel == "cauchy"    :
            c = 1.0+r2/ (s*s)
            return - 1 / (s*s*c*c)
        return -1

    def np_kernel_sphere(nalpha, nbeta, s, kernel = "binet") :
        prs = nalpha @ nbeta.T
        if kernel == "binet"                 : return prs**2
        elif kernel == "linear"              : return prs
        elif kernel == "gaussian_unoriented" : return np.exp( (-2.0 + 2.0 * prs*prs) / (s*s))
        elif kernel == "gaussian_oriented"   : return np.exp( (-2.0 + 2.0 * prs) / (s*s))
        return -1

    if True:
        x     = np.random.rand(sizeX,dimPoint).astype('float32')
        y     = np.random.rand(sizeY,dimPoint).astype('float32')
        f     = np.random.rand(sizeX,dimSig).astype('float32')
        g     = np.random.rand(sizeY,dimSig).astype('float32')
        alpha = np.random.rand(sizeX,dimVect ).astype('float32')
        beta  = np.random.rand(sizeY,dimVect ).astype('float32')
    else :
        x     =(np.linspace(0.5,2,sizeX)[:,np.newaxis] * np.linspace(0,3,dimPoint)[np.newaxis,:] ).astype('float32')
        alpha =(np.linspace(-1.1,2,sizeX)[:,np.newaxis] * np.linspace(1,1,dimPoint)[np.newaxis,:]).astype('float32')
        f     =-(np.linspace(1,2,sizeX)[:,np.newaxis] ).astype('float32')
        y     =-(np.linspace(1,3,sizeY)[:,np.newaxis] * np.linspace(-1,1,dimPoint)[np.newaxis,:] ).astype('float32')
        beta  =-(np.linspace(1,2,sizeY)[:,np.newaxis] * np.linspace(1,2,dimPoint)[np.newaxis,:]).astype('float32')
        g     =(np.linspace(1,2,sizeY)[:,np.newaxis] ).astype('float32')
    
    kgeom = "gaussian"
    ksig = "gaussian"
    ksphere = "linear"
    
    # Call cuda kernel
    gamma = np.zeros(sizeX*dimPoint).astype('float32')
    cuda_shape_scp_dx(x, y, f, g, alpha, beta, gamma, sigma_geom, sigma_sig, sigma_var,kgeom,ksig,ksphere)

    
    # Python version
    areaa = np.linalg.norm(alpha,axis=1)
    areab = np.linalg.norm(beta,axis=1)

    nalpha = alpha / areaa[:,np.newaxis]
    nbeta = beta / areab[:,np.newaxis]
    
    dxy =  (areaa[:,np.newaxis] * areab[np.newaxis,:]) * dnp_kernel(x,y,sigma_geom,kgeom) *  np_kernel(f,g,sigma_sig,ksig) * np_kernel_sphere(nalpha,nbeta,sigma_var,ksphere)

    gamma_py = np.ndarray(shape=(sizeX,dimPoint) ).astype('float32')
    for i in range(dimPoint):
        gamma_py[:,i] =  2*np.sum(  dxy * ((x[:,np.newaxis,:] - y[np.newaxis,:,:])[:,:,i]) , axis=1 )

    # compare output
    print("\ngradX Fshape distance cuda:")
    print(gamma)

    print("\ngradX Fshape distance numpy:")
    print(gamma_py)

    print("\nIs everything okay ? ")
    print(np.allclose(gamma.flatten(), gamma_py.flatten() ))
