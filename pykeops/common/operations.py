from pykeops.common.utils import get_tools

def Genred_common(binding, formula, aliases, reduction_op, axis, cuda_type, opt_arg, formula2):
    tools = get_tools(binding)
    if reduction_op=='SoftMax':
        reduction_op_internal = 'LogSumExpVect'
        formula2 = 'Concat(IntCst(1),' + formula2 + ')'
    elif reduction_op=='LogSumExp' and formula2:
        reduction_op_internal = 'LogSumExpVect'
    else:
        reduction_op_internal = reduction_op        
    my_routine = tools.Genred_lowlevel(formula, aliases, reduction_op_internal, axis, cuda_type, opt_arg, formula2)
    def f(*args, backend='auto', device_id=-1):
        out = my_routine(*args, backend=backend, device_id=device_id)
        if reduction_op=='SoftMax':
            out = out[:,2:]/out[:,1][:,None]
        elif reduction_op=='LogSumExp':
            if out.shape[1]==2: # means (m,s) with m scalar and s scalar
                out = tools.view(out[:,0] + tools.log(out[:,1]),(-1,1))
            else: # here out.shape[1]>2, means (m,s) with m scalar and s vectorial
                out = out[:,0][:,None] + tools.log(out[:,1:])
        return out
    return f

def ConjugateGradientSolver(binding,linop,b,eps=1e-6):
    # Conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    tools = get_tools(binding)
    delta = tools.size(b)*eps**2
    a = 0
    r = tools.copy(b)
    nr2 = (r**2).sum()
    if nr2 < delta:
        return 0*r
    p = tools.copy(r)
    k = 0
    while True:
        Mp = linop(p)
        alpha = nr2/(p*Mp).sum()
        a += alpha*p
        r -= alpha*Mp
        nr2new = (r**2).sum()
        if nr2new < delta:
            break
        p = r + (nr2new/nr2)*p
        nr2 = nr2new
        k += 1
    return a
    

def KernelLinearSolver(binding,K,x,b,lmbda=0,eps=1e-6,precond=False,precondKernel=None):
    
    tools = get_tools(binding)
            
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
