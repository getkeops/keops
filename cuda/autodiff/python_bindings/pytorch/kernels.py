#import os.path
#import sys
#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '')

from .kernel_product_generic import GenericKernelProduct
from .logsumexp_generic import GenericLogSumExp


def GaussianKernel( c, x, y, b, mode = "standard") :
    if mode == "standard" : 
        genconv  = GenericKernelProduct().apply
        dimpoint = x.size(1) ; dimout = b.size(1)
        
        aliases  = ["DIMPOINT = "+str(dimpoint), "DIMOUT = "+str(dimout),
                    "C = Param<0>"          ,   # 1st parameter
                    "X = Var<0,DIMPOINT,0>" ,   # 1st variable, dim DIM,    indexed by i
                    "Y = Var<1,DIMPOINT,1>" ,   # 2nd variable, dim DIM,    indexed by j
                    "B = Var<2,DIMOUT  ,1>" ]   # 3rd variable, dim DIMOUT, indexed by j
        #   R   =        exp(            C    *   -          |         X-Y|^2   )*  B
        formula = "Scal< Exp< Scal<Constant<C>, Minus<SqNorm2<Subtract<X,Y>>> > >,  B>"
        # stands for:     R_i   ,   C  ,      X_i    ,      Y_j    ,     B_j    .
        signature = [ (dimout,0), (1,2), (dimpoint,0), (dimpoint,1), (dimout,1) ]
        sum_index = 0 # the output vector is indexed by "i" (CAT=0)
        return genconv( aliases, formula, signature, sum_index, c, x, y, b )
    
    elif mode == "log" :
        genconv  = GenericLogSumExp().apply
        dimpoint = x.size(1) ; dimout = b.size(1)
        
        if not dimout == 1 : 
            raise ValueError("As of today, LogSumExp has only been implemented for scalar-valued functions.")

        aliases  = ["DIMPOINT = "+str(dimpoint), "DIMOUT = "+str(dimout),
                    "C = Param<0>"          ,   # 1st parameter
                    "X = Var<0,DIMPOINT,0>" ,   # 1st variable, dim DIM,    indexed by i
                    "Y = Var<1,DIMPOINT,1>" ,   # 2nd variable, dim DIM,    indexed by j
                    "V = Var<2,DIMOUT  ,1>" ]   # 3rd variable, dim DIMOUT, indexed by j
                    
        # stands for:     R_i   ,   C  ,      X_i    ,      Y_j    ,     V_j    .
        signature = [ (dimout,0), (1,2), (dimpoint,0), (dimpoint,1), (dimout,1) ]
        #   R   =                     C *   -          |         X-Y|^2   +  V
        formula = "Add< Scal<Constant<C>, Minus<SqNorm2<Subtract<X,Y>>> > ,  V>"
        
        sum_index = 0 # the output vector is indexed by "i" (CAT=0)
        return genconv( aliases, formula, signature, sum_index, c, x, y, b )

    else :
        raise ValueError("'mode' should either be 'standard' or 'log'.")

def TwoGaussiansKernel( c1, c2, x, y, b, mode = "standard") :
    if mode == "standard" : 
        genconv  = GenericKernelProduct().apply
        dimpoint = x.size(1) ; dimout = b.size(1)
        
        aliases  = ["DIMPOINT = "+str(dimpoint), "DIMOUT = "+str(dimout),
                    "C1 = Param<0>"          ,   # 1st parameter
                    "C2 = Param<1>"          ,   # 1st parameter
                    "X = Var<0,DIMPOINT,0>" ,   # 1st variable, dim DIM,    indexed by i
                    "Y = Var<1,DIMPOINT,1>" ,   # 2nd variable, dim DIM,    indexed by j
                    "B = Var<2,DIMOUT  ,1>" ]   # 3rd variable, dim DIMOUT, indexed by j
        #   R   =            exp(            C    *   -          |         X-Y|^2   )*  B
        formula = "Scal< Add<Exp< Scal<Constant<C1>, Minus<SqNorm2<Subtract<X,Y>>> > >," \
                + "Exp< Scal<Constant<C2>, Minus<SqNorm2<Subtract<X,Y>>> > > >" \
                +       ",  B>"
        # stands for:     R_i   ,   C1  ,    C2 ,      X_i    ,      Y_j    ,     B_j    .
        signature = [ (dimout,0),  (1,2),  (1,2), (dimpoint,0), (dimpoint,1), (dimout,1) ]
        sum_index = 0 # the output vector is indexed by "i" (CAT=0)
        return genconv( aliases, formula, signature, sum_index, c1, c2, x, y, b )
    
    elif mode == "log" :
        raise NotImplementedError()








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



































