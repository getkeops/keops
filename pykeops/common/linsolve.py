def ConjugateGradientSolver(backend,linop,b,eps=1e-6):
    # Conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    if backend == 'numpy':
        from pykeops.numpy.utils import numpytools
        tools = numpytools()
    elif backend == 'torch':
        from pykeops.torch.utils import torchtools
        tools = torchtools()
    a = 0
    r = tools.copy(b)
    nr2 = (r**2).sum()
    if nr2 < eps**2:
        return 0*r
    p = tools.copy(r)
    k = 0
    while True:
        Mp = linop(p)
        alpha = nr2/(p*Mp).sum()
        a += alpha*p
        r -= alpha*Mp
        nr2new = (r**2).sum()
        if nr2new < eps**2:
            break
        p = r + (nr2new/nr2)*p
        nr2 = nr2new
        k += 1
    print("numiters=",k)
    return a
    

def KernelLinearSolver(tools,K,x,b,lmbda=0,eps=1e-6,precond=False,precondKernel=None):
    
    tools.set_types(x)
            
    def PreconditionedConjugateGradientSolver(linop,b,invprecondop,eps=1e-6):
        # Preconditioned conjugate gradient algorithm to solve linear system of the form
        # Ma=b where linop is a linear operation corresponding
        # to a symmetric and positive definite matrix
        # invprecondop is linear operation corresponding to the inverse of the preconditioner matrix
        a = 0
        r = tools.copy(b)
        z = invprecondop(r)
        p = tools.copy(z)
        rz = (r*z).sum()
        k = 0
        while True:    
            alpha = rz/(p*linop(p)).sum()
            a += alpha*p
            r -= alpha*linop(p)
            if (r**2).sum() < eps**2:
                break
            z = invprecondop(r)
            rznew = (r*z).sum()
            p = z + (rznew/rz)*p
            rz = rznew
            k += 1
        return a

    def NystromInversePreconditioner(K,Kspec,x,lmbda):
        N,D = x.shape
        m = int(np.sqrt(N))
        ind = np.random.choice(range(N),m,replace=False)
        u = x[ind,:]
        M = K(u,u) + Kspec(tools.tile(u,(m,1)),tools.tile(u,(1,m)).reshape(-1,D),x).reshape(m,m)
        def invprecondop(r):
            a = tools.solve(M,K(u,x,r))
            return (r - K(x,u,a))/lmbda
        return invprecondop

    def KernelLinOp(a):
        return K(x,x,a) + lmbda*a
        
    def GaussKernel(D,Dv,sigma):
        formula = 'Exp(-oos2*SqDist(x,y))*b'
        variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                     'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
                     'b = Vy(' + str(Dv) + ')',  # Third arg  : j-variable, of size Dv
                     'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
        my_routine = tools.Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=tools.dtype)
        oos2 = tools.array([1.0/sigma**2])
        KernelMatrix = GaussKernelMatrix(sigma)
        def K(x,y,b=None):
            if b is None:
                return KernelMatrix(x,y)
            else:
                return my_routine(x,y,b,oos2)
        return K

    def GaussKernelNystromPrecond(D,sigma):
        formula = 'Exp(-oos2*(SqDist(u,x)+SqDist(v,x)))'
        variables = ['u = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                     'v = Vx(' + str(D) + ')',  # Second arg  : i-variable, of size D
                     'x = Vy(' + str(D) + ')',  # Third arg  : j-variable, of size D
                     'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
        my_routine = tools.Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=tools.dtype)
        oos2 = tools.array([1.0/sigma**2])
        KernelMatrix = GaussKernelMatrix(sigma)
        def K(u,v,x):
            return my_routine(u,v,x,oos2)
        return K

    def GaussKernelMatrix(sigma):
        oos2 = 1.0/sigma**2
        def f(x,y):
            D = x.shape[1]
            sqdist = 0
            for k in range(D):
                sqdist += (x[:,k][:,None]-tools.transpose(y[:,k][:,None]))**2
            return tools.exp(-oos2*sqdist)
        return f
    
    if type(K)==tuple:
        if K[0]=="gaussian":
            D = K[1]
            Dv = K[2]
            sigma = K[3]
            K = GaussKernel(D,Dv,sigma)
            if precond:
                precondKernel = GaussKernelNystromPrecond(D,sigma)        

    if precond:
        invprecondop = NystromInversePreconditioner(K,precondKernel,x,lmbda)
        a = PreconditionedConjugateGradientSolver(KernelLinOp,b,invprecondop,eps)
    else:
        a = ConjugateGradientSolver(tools,KernelLinOp,b,eps=eps)
        
    return a
