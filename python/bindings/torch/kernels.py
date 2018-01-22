#import os.path
#import sys
#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '')

from .kernel_product_generic import GenericKernelProduct
from .logsumexp_generic import GenericLogSumExp

def _squared_distances(x, y) :
    x_i = x.unsqueeze(1)         # Shape (N,D) -> Shape (N,1,D)
    y_j = y.unsqueeze(0)         # Shape (M,D) -> Shape (1,M,D)
    return ((x_i-y_j)**2).sum(2) # N-by-M matrix, xmy[i,j] = |x_i-y_j|^2

def _radial_kernel(routine, x, y, b) :
    K = routine(_squared_distances(x,y))
    return K @ b  # Matrix product between the Kernel operator and the source field b

def _log_sum_exp(mat, dim):
    """
    Computes the log-sum-exp of a matrix with a numerically stable scheme, 
    in the user-defined summation dimension: exp is never applied
    to a number >= 0, and in each summation row, there is at least
    one "exp(0)" to stabilize the sum.
    
    For instance, if dim = 1 and mat is a 2d array, we output
                log( sum_j exp( mat[i,j] )) 
    by factoring out the row-wise maximas.
    """
    max_rc = torch.max(mat, dim)[0]
    return max_rc + torch.log(torch.sum(torch.exp(mat - max_rc.unsqueeze(dim)), dim))

def _radial_kernel_log(routine, x, y, b_log) :
    """
    Computes log( K(x_i,y_j) @ b_j) = log( \sum_j k(x_i-y_j) * b_j) in the log domain,
    where k is a kernel function speciefied by "routine".
    """
    C = routine(_squared_distances(x,y))
    return _log_sum_exp( C + b_log.view(1,-1) , 1 ).view(-1,1) 

def RadialKernel( formula, routine, gamma, x, y, b, mode = "sum", backend="auto") :
    if backend == "pytorch" :
        if   mode == "sum" : return     _radial_kernel(routine, x, y, b)
        elif mode == "log" : return _radial_kernel_log(routine, x, y, b)
        else : raise ValueError('"mode" should either be "sum" or "log".')

    else :
        if   mode == "sum" : genconv  = GenericKernelProduct().apply
        elif mode == "log" : genconv  = GenericLogSumExp().apply
        else : raise ValueError('"mode" should either be "sum" or "log".')
        
        dimpoint = x.size(1) ; dimout = b.size(1)
        
        aliases  = ["DIMPOINT = "+str(dimpoint), "DIMOUT = "+str(dimout),
                    "G = Param<0>"          ,   # 1st parameter
                    "X = Var<0,DIMPOINT,0>" ,   # 1st variable, dim DIM,    indexed by i
                    "Y = Var<1,DIMPOINT,1>" ,   # 2nd variable, dim DIM,    indexed by j
                    "B = Var<2,DIMOUT  ,1>" ]   # 3rd variable, dim DIMOUT, indexed by j

        # stands for:     R_i   ,   G  ,      X_i    ,      Y_j    ,     B_j    .
        signature = [ (dimout,0), (1,2), (dimpoint,0), (dimpoint,1), (dimout,1) ]
        sum_index = 0 # the output vector is indexed by "i" (CAT=0)
        return genconv( backend, aliases, formula, signature, sum_index, gamma, x, y, b )
    

def GaussianKernel( gamma, x, y, b, mode = "sum", backend = "auto") :
    if   mode=="sum": 
        formula = "Scal< Exp< Scal<Constant<G>, Minus<SqNorm2<Subtract<X,Y>>> > >,  B>"
        routine = lambda xmy2 : (-gamma*xmy2).exp()
    elif mode=="log": 
        formula = "Add< Scal<Constant<G>, Minus<SqNorm2<Subtract<X,Y>>> > ,  B>"
        routine = lambda xmy2 :  -gamma*xmy2
    else : raise ValueError('"mode" should either be "sum" or "log".')
    return RadialKernel( formula, routine, gamma, x, y, b, mode, backend)


"""
    elif mode == "laplace"  : K = torch.exp( - torch.sqrt(xmy + (s**2)) )
    elif mode == "energy"   : K = torch.pow(   xmy + (s**2), -.25 )

    if   mode == "gaussian"     : C =  - xmy / (s**2) 
    elif mode == "laplace"      : C =  - torch.sqrt(xmy + (s**2)) 
    elif mode == "energy"       : C =  -.25 * torch.log( xmy + (s**2) )
    elif mode == "exponential"  : C =  - torch.sqrt(xmy) / s 
"""


def StandardKernelProduct(gamma, x,y,b, name, mode, backend = "auto") :
    """
    Convenience function, providing the standard formulas implemented
    in the libkp library.

    Computes K(x_i,y_j) @ nu_j = \sum_j k(x_i-y_j) * nu_j
    where k is a kernel function specified by "name".

    In the simplest cases, the signature of our function is as follow:
    Args:
        x ( (N,D) torch Variable) : sampling point cloud 
        y ( (M,D) torch Variable) : source point cloud
        b ( (M,E) torch Variable) : source vector field, supported by y
        params (dict)			  : convenient way of storing the kernel's parameters
        mode   (string)			  : either "sum" (for classical summation) 
                                    or "log" (outputs the 'log' of "sum", computed in a
                                    numerically robust way)

    Returns:
        v ( (N,E) torch Variable) : sampled vector field 
                                    (or its coordinate-wise logarithm, if mode=="log")

    However, if, say, params["formula"]=="gaussian currents",
    we use a kernel function parametrized by locations
    X_i, Y_j AND directions U_i, V_j.
    The argument "x" is then a pair (X,U) of (N,D) torch Variables,
    while        "y" is      a pair (Y,V) of (M,E) torch Variables.

    Possible values for "name" are:
        - "gaussian"

    N.B.: The backend is specified in params["backend"], and its value
        may have a critical influence on performance:
    - "pytorch" means that we use a naive "matrix-like", full-pytorch formula.
                It does not scale well as soon as N or M > 5,000.
    - "CPU"     means that we use a CPU implementation of the libkp C++ routines.
                It performs okay-ish, but cannot rival GPU implementations.
    - "GPU_2D"  means that we use a GPU implementation of the libkp C++/CUDA routines,
                with a 2-dimensional job distribution scheme. 
                It may come useful if, for instance, N < 200 and 10,000 < M. 
    - "GPU_1D"  means that we use a GPU implementation of the libkp C++/CUDA routines,
                with a 1-dimensional distribution scheme (one thread = one line of x).
                If you own an Nvidia GPU, this is the go-to method for large point clouds. 

    If the backend is not specified or "auto", the libkp routines will try to
    use a suitable one depending on your configuration + the dimensions of x, y and b.
    """

    point_kernels           = { "gaussian"         : GaussianKernel }
    point_direction_kernels = { }#"gaussian current" : GaussianCurrentKernel} 

    if   name in point_kernels :
        return point_kernels[name]( gamma, x, y, b, mode = mode, backend = backend)
    elif name in point_direction_kernels :
        X,U = x; Y,V = y
        return point_direction_kernels[name]( gamma, X, Y, U, V, b, mode = mode, backend = backend)
    else :
        raise NotImplementedError("Kernel name '"+name+"'. "\
                                 +'Available values are "' + '", "'.join(point_kernels.keys()) \
                                 + '", "' + '", "'.join(point_direction_kernels.keys())+'".' )





if __name__ == "__main__" :
    import numpy as np
    import torch
    from   torch.autograd import Variable
    use_cuda = False#torch.cuda.is_available()
    dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    def kernel_product_pytorch(x,y,nu, mode = "gaussian", s = 1.) :
        """
        Computes K(x_i,y_j) @ nu_j = \sum_j k(x_i-y_j) * nu_j
        where k is a kernel function (say, a Gaussian) of deviation s.
        """
        x_i = x.unsqueeze(1)        # Shape (N,d) -> Shape (N,1,d)
        y_j = y.unsqueeze(0)        # Shape (M,d) -> Shape (1,M,d)
        xmy = ((x_i-y_j)**2).sum(2) # N-by-M matrix, xmy[i,j] = |x_i-y_j|^2
        if   mode == "gaussian" : K = torch.exp( - xmy / (s**2) )
        elif mode == "twogaussians" : K = torch.exp( - xmy / (s[0]**2) ) + torch.exp( - xmy / (s[1]**2))
        elif mode == "laplace"  : K = torch.exp( - torch.sqrt(xmy + (s**2)) )
        elif mode == "energy"   : K = torch.pow(   xmy + (s**2), -.25 )
        return K @ (nu.view(-1,1))  # Matrix product between the Kernel operator and the vector nu.
    
    def kernel_product_libkp(x,y,nu, mode = "gaussian", s = 1.) :
        if mode == "gaussian" :
            c  = Variable(torch.Tensor([1/(s**2)])).type(dtype)
            return GaussianKernel( c, x, y, nu.view(-1,1))
        elif mode == "twogaussians" :
            c1 = Variable(torch.Tensor([1/(s[0]**2)])).type(dtype)
            c2 = Variable(torch.Tensor([1/(s[1]**2)])).type(dtype)
            return TwoGaussiansKernel( c1, c2, x, y, nu.view(-1,1))
        else :
            raise NotImplementedError()
    if False :
        def log_sum_exp(mat, dim):
            """
            Computes the log-sum-exp of a matrix with a numerically stable scheme, 
            in the user-defined summation dimension: exp is never applied
            to a number >= 0, and in each summation row, there is at least
            one "exp(0)" to stabilize the sum.
            
            For instance, if dim = 1 and mat is a 2d array, we output
                        log( sum_j exp( mat[i,j] )) 
            by factoring out the row-wise maximas.
            """
            max_rc = torch.max(mat, dim)[0]
            return max_rc + torch.log(torch.sum(torch.exp(mat - max_rc.unsqueeze(dim)), dim))
        
        def kernel_product_log_pytorch(x,y,nu_log, mode = "gaussian", s = 1.) :
            """
            Computes log( K(x_i,y_j) @ nu_j) = log( \sum_j k(x_i-y_j) * nu_j) in the log domain,
            where k is a kernel function (say, a Gaussian) of deviation s.
            """
            x_i = x.unsqueeze(1)        # Shape (N,d) -> Shape (N,1,d)
            y_j = y.unsqueeze(0)        # Shape (M,d) -> Shape (1,M,d)
            xmy = ((x_i-y_j)**2).sum(2) # N-by-M matrix, xmy[i,j] = |x_i-y_j|^2
            if   mode == "gaussian"     : C =  - xmy / (s**2) 
            elif mode == "laplace"      : C =  - torch.sqrt(xmy + (s**2)) 
            elif mode == "energy"       : C =  -.25 * torch.log( xmy + (s**2) )
            elif mode == "exponential"  : C =  - torch.sqrt(xmy) / s 
            return log_sum_exp( C + nu_log.view(1,-1) , 1 ).view(-1,1) # Matrix product between the Kernel operator and the vector nu.

        def kernel_product_log_libkp(x,y,nu_log, mode = "gaussian", s = 1.) :
            c = Variable(torch.Tensor([1/(s**2)])).type(dtype)
            if mode == "gaussian" :
                return GaussianKernel( c, x, y, nu_log.view(-1,1), mode = "log")
            else :
                raise NotImplementedError()

    def kernel_product(x,y,nu,mode="gaussian",s=1.,backend="pytorch") :
        if backend == "pytorch" :
            return kernel_product_pytorch(x,y,nu, mode, s)
        elif backend == "libkp" :
            return kernel_product_libkp(x,y,nu, mode, s)
    
    if False :
        def kernel_product_log(x,y,nu,mode="gaussian",s=1.,backend="pytorch") :
            if backend == "pytorch" :
                return kernel_product_log_pytorch(x,y,nu, mode, s)
            elif backend == "libkp" :
                return kernel_product_log_libkp(x,y,nu, mode, s)

    N = 100000 ; M = 100000 ; D = 2 ; E = 1

    e = .6 * torch.normal( means = torch.zeros(M,D) ).type(dtype)
    e = torch.autograd.Variable(e, requires_grad = True)

    a = .6 * torch.normal( means = torch.zeros(N,E) ).type(dtype)
    a = torch.autograd.Variable(a, requires_grad = True)

    x = .6 * torch.normal( means = torch.zeros(N,D) ).type(dtype) 
    x = torch.autograd.Variable(x, requires_grad = True)

    y = .2 * torch.normal( means = torch.zeros(M,D) ).type(dtype) 
    y = torch.autograd.Variable(y, requires_grad = True)

    b = .6 * torch.normal( means = torch.zeros(M,E) ).type(dtype) 
    b = torch.autograd.Variable(b, requires_grad = True)

    #s = torch.Tensor([1.778]).type(dtype)
    #s = torch.autograd.Variable(s, requires_grad = False)
    s = 1.778
    s = [10., 1.]

    #--------------------------------------------------#
    # check the class KernelProduct
    #--------------------------------------------------#
    for backend in ["libkp"] :
        print('\n\nTest with the '+backend+' backend :-----------------\n')
        Kxy_b = kernel_product(x,y,b, "twogaussians", s = s, backend=backend)
        print("kernel product : ", Kxy_b[-5:,:])

        Hxy_b     = (Kxy_b**2).sum()

        if True :
            def Ham(q,p) :
                Kq_p  = kernel_product(q,q,p, "twogaussians", s=s, backend=backend)
                return torch.dot( p.view(-1), Kq_p.view(-1) )
            ham0 = Ham(y, b)
            
            print("Ham0:") ; print(ham0)

        if True :
            grad_x = torch.autograd.grad(Kxy_b,x,a,create_graph = True)[0]
            grad_y = torch.autograd.grad(Kxy_b,y,a,create_graph = True)[0]
            grad_b = torch.autograd.grad(Kxy_b,b,a,create_graph = True)[0]
            grad_yb = torch.autograd.grad(grad_y,b,e, create_graph = True)[0]
            
            print('grad_x   :\n', grad_x[:5,:].data.cpu().numpy())
            print('grad_y   :\n', grad_y[:5,:].data.cpu().numpy())
            print('grad_b   :\n', grad_b[:5,:].data.cpu().numpy())
            print('grad_yb  :\n', grad_yb[:5,:].data.cpu().numpy())

        if False :
            grad_x = torch.autograd.grad(Hxy_b,x,create_graph = True)[0]
            grad_y = torch.autograd.grad(Hxy_b,y,create_graph = True)[0]
            grad_b = torch.autograd.grad(Hxy_b,b,create_graph = True)[0]
            grad_yb = torch.autograd.grad(grad_y,b, torch.ones(grad_y.size()).type(dtype), create_graph = True)[0]
            
            print('grad_x   :\n', grad_x[:5,:].data.cpu().numpy())
            print('grad_y   :\n', grad_y[:5,:].data.cpu().numpy())
            print('grad_b   :\n', grad_b[:5,:].data.cpu().numpy())
            print('grad_yb  :\n', grad_yb[:5,:].data.cpu().numpy())

    if False :
        for backend in ["pytorch", "libkp"] :
            print('\n\nTest with the '+backend+' backend :-----------------\n')
            Kxy_b = kernel_product(x,y,b, "gaussian", s = s, backend=backend)
            print("kernel product : ", Kxy_b[-5:,:])

            Kxy_b_log = kernel_product_log(x,y,b, "gaussian", s=s, backend=backend)
            print("kernel product log : ", Kxy_b_log[:5,:])

            Hxy_b     = (Kxy_b**2).sum()
            Hxy_b_log = (Kxy_b_log**2).sum()

            if True :
                def Ham(q,p) :
                    Kq_p  = kernel_product(q,q,p, "gaussian", s=s, backend=backend)
                    return torch.dot( p.view(-1), Kq_p.view(-1) )
                ham0 = Ham(y, b)
                
                print("Ham0:") ; print(ham0)

            if True :
                grad_x = torch.autograd.grad(Kxy_b,x,a,create_graph = True)[0]
                grad_y = torch.autograd.grad(Kxy_b,y,a,create_graph = True)[0]
                grad_b = torch.autograd.grad(Kxy_b,b,a,create_graph = True)[0]
                grad_yb = torch.autograd.grad(grad_y,b,e, create_graph = True)[0]
                
                print('grad_x   :\n', grad_x[:5,:].data.cpu().numpy())
                print('grad_y   :\n', grad_y[:5,:].data.cpu().numpy())
                print('grad_b   :\n', grad_b[:5,:].data.cpu().numpy())
                print('grad_yb  :\n', grad_yb[:5,:].data.cpu().numpy())

            if False :
                grad_x = torch.autograd.grad(Hxy_b,x,create_graph = True)[0]
                grad_y = torch.autograd.grad(Hxy_b,y,create_graph = True)[0]
                grad_b = torch.autograd.grad(Hxy_b,b,create_graph = True)[0]
                grad_yb = torch.autograd.grad(grad_y,b, torch.ones(grad_y.size()).type(dtype), create_graph = True)[0]
                
                print('grad_x   :\n', grad_x[:5,:].data.cpu().numpy())
                print('grad_y   :\n', grad_y[:5,:].data.cpu().numpy())
                print('grad_b   :\n', grad_b[:5,:].data.cpu().numpy())
                print('grad_yb  :\n', grad_yb[:5,:].data.cpu().numpy())



































