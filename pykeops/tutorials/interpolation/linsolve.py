
import time
import numpy as np
import torch
from pykeops.numpy import Genred as GenredNumpy
from pykeops.torch import Genred as GenredTorch
from pykeops.torch.generic.generic_red import GenredAutograd
from pykeops import default_cuda_type
from pykeops.common.utils import axis2cat, cat2axis
from pykeops.common.parse_type import get_type, get_sizes, complete_aliases
from pykeops.common.get_options import get_tag_backend
from pykeops.common.keops_io import load_keops

include_dirs = torch.utils.cpp_extension.include_paths()[0]

def ConjugateGradientSolver(linop,b,eps=1e-6):
    # Conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    arraytype = type(b)
    copy = np.copy if arraytype == np.ndarray else torch.clone
    a = 0
    r = copy(b)
    nr2 = (r**2).sum()
    if nr2 < eps**2:
        return 0*r
    p = copy(r)
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

       
        





class InvKernelOpAutograd(torch.autograd.Function):
    """
    This class is the entry point to pytorch auto grad engine.
    """

    @staticmethod
    def forward(ctx, formula, aliases, varinvalias, backend, cuda_type, device_id, *args):

        myconv = load_keops(formula, aliases, cuda_type, 'torch', ['-DPYTORCH_INCLUDE_DIR=' + include_dirs])

        # Context variables: save everything to compute the gradient:
        ctx.formula = formula
        ctx.aliases = aliases
        ctx.varinvalias = varinvalias
        ctx.backend = backend
        ctx.cuda_type = cuda_type
        ctx.device_id = device_id
        ctx.myconv = myconv

        tmp = aliases.copy()
        for (i,s) in enumerate(tmp):
            tmp[i] = s[:s.find("=")].strip()
        varinvpos = tmp.index(varinvalias)
        varinv = args[varinvpos]
        ctx.varinvpos = varinvpos

        nx, ny = get_sizes(aliases, *args)

        tagCPUGPU, tag1D2D, tagHostDevice = get_tag_backend(backend, args)

        if tagCPUGPU==1 & tagHostDevice==1:
            device_id = args[0].device.index
            for i in range(1,len(args)):
                if args[i].device.index != device_id:
                    raise ValueError("[KeOps] Input arrays must be all located on the same device.")

        def linop(var):
            newargs = args[:varinvpos] + (var,) + args[varinvpos+1:]
            return myconv.genred_pytorch(nx, ny, tagCPUGPU, tag1D2D, tagHostDevice, device_id, *newargs)

        result = ConjugateGradientSolver(linop,varinv.data,eps=1e-16)

        # relying on the 'ctx.saved_variables' attribute is necessary  if you want to be able to differentiate the output
        #  of the backward once again. It helps pytorch to keep track of 'who is who'.
        ctx.save_for_backward(*args,result)

        return result

    @staticmethod
    def backward(ctx, G):
        print("in InvKernelOpAutograd.backward")
        formula = ctx.formula
        aliases = ctx.aliases
        varinvalias = ctx.varinvalias
        backend = ctx.backend
        cuda_type = ctx.cuda_type
        device_id = ctx.device_id
        myconv = ctx.myconv
        varinvpos = ctx.varinvpos
        args = ctx.saved_tensors[:-1]  # Unwrap the saved variables
        nargs = len(args)
        result = ctx.saved_tensors[-1].detach()

        # If formula takes 5 variables (numbered from 0 to 4), then the gradient
        # wrt. the output, G, should be given as a 6-th variable (numbered 5),
        # with the same dim-cat as the formula's output.
        eta = 'Var(' + str(nargs) + ',' + str(myconv.dimout) + ',' + str(myconv.tagIJ) + ')'
      
        # there is also a new variable for the formula's output
        resvar = 'Var(' + str(nargs+1) + ',' + str(myconv.dimout) + ',' + str(myconv.tagIJ) + ')'
        
        newargs = args[:varinvpos] + (G,) + args[varinvpos+1:]
        KinvG = InvKernelOpAutograd.apply(formula, aliases, varinvalias, backend, cuda_type, device_id, *newargs)

        grads = []  # list of gradients wrt. args;

        for (var_ind, sig) in enumerate(aliases):  # Run through the arguments
            # If the current gradient is to be discarded immediatly...
            if not ctx.needs_input_grad[var_ind + 6]:  # because of (formula, aliases, varinvalias, backend, cuda_type, device_id)
                grads.append(None)  # Don't waste time computing it.

            else:  # Otherwise, the current gradient is really needed by the user:

                if var_ind == varinvpos:
                    grads.append(KinvG)
                else:
                    # adding new aliases is way too dangerous if we want to compute
                    # second derivatives, etc. So we make explicit references to Var<ind,dim,cat> instead.
                    # New here (Joan) : we still add the new variables to the list of "aliases" (without giving new aliases for them)
                    # these will not be used in the C++ code, 
                    # but are useful to keep track of the actual variables used in the formula
                    _, cat, dim, pos = get_type(sig, position_in_list=var_ind)
                    var = 'Var(' + str(pos) + ',' + str(dim) + ',' + str(cat) + ')'  # V
                    formula_g = 'Grad_WithSavedForward(' + formula + ', ' + var + ', ' + eta + ', ' + resvar + ')'  # Grad<F,V,G,R>
                    aliases_g = aliases + [eta, resvar]
                    args_g = args[:varinvpos] + (result,) + args[varinvpos+1:] + (-KinvG,) + (result,)  # Don't forget the gradient to backprop !

                    # N.B.: if I understand PyTorch's doc, we should redefine this function every time we use it?
                    genconv = GenredAutograd().apply

                    if cat == 2:  # we're referring to a parameter, so we'll have to sum both wrt 'i' and 'j'
                        # WARNING !! : here we rely on the implementation of DiffT in files in folder keops/core/reductions
                        # if tagI==cat of V is 2, then reduction is done wrt j, so we need to further sum output wrt i
                        grad = genconv(formula_g, aliases_g, backend, cuda_type, device_id, *args_g)
                        # Then, sum 'grad' wrt 'i' :
                        # I think that '.sum''s backward introduces non-contiguous arrays,
                        # and is thus non-compatible with GenredAutograd: grad = grad.sum(0)
                        # We replace it with a 'handmade hack' :
                        grad = torch.ones(1, grad.shape[0]).type_as(grad.data) @ grad
                        grad = grad.view(-1)
                    else:
                        grad = genconv(formula_g, aliases_g, backend, cuda_type, device_id, *args_g)
                    grads.append(grad)
         
        # Grads wrt. formula, aliases, backend, cuda_type, device_id, *args
        return (None, None, None, None, None, None, *grads)



class InvKernelOp:
    """
    Note: Class should not inherit from GenredAutograd due to pickling errors
    """
    def __init__(self, formula, aliases, varinvalias, reduction_op='Sum', axis=0, cuda_type=default_cuda_type):
        self.formula = reduction_op + 'Reduction(' + formula + ',' + str(axis2cat(axis)) + ')'
        self.aliases = complete_aliases(formula, list(aliases)) # just in case the user provided a tuple
        self.varinvalias = varinvalias
        self.cuda_type = cuda_type

    def __call__(self, *args, backend='auto', device_id=-1):
        return InvKernelOpAutograd.apply(self.formula, self.aliases, self.varinvalias, backend, self.cuda_type, device_id, *args)












        
class InvLinOp_Impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, linop, b, *args):
        print("InvLinOp_Impl forward 1")
        # solves b=linop(a) where linop is a linear operation
        # args is optional and used only in backward : it gives the variables
        # wrt which we will compute the gradient
        ctx.linop = linop
        a = ConjugateGradientSolver(linop,b.data,eps=1e-16)
        ctx.save_for_backward(a,*args)
        return a
    @staticmethod
    def backward(ctx, grad_output):
        # gradients of a wrt variables b and *args
        linop = ctx.linop
        a,*args = ctx.saved_tensors
        e = InvLinOp(linop,grad_output) # this gives gradient wrt b
        #with torch.enable_grad():
        #    grads = torch.autograd.grad(linop(a.data),args,-e,create_graph=True)
        abar = torch.tensor(a.data, requires_grad=True)
        grads = []
        with torch.enable_grad():
            for x in args:
                grads.append(specfun(torch.autograd.grad(linop(abar),x,-e,create_graph=True)[0],abar,a))
        return (None, e, *grads)

class specfun_Impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx,f,abar,a):
        ctx.save_for_backward(f,abar,a)
        return f.data
    @staticmethod
    def backward(ctx, grad_output):
        f,abar,a = ctx.saved_tensors
        with torch.enable_grad():
            gfa = specfun(torch.autograd.grad(f,abar,grad_output,create_graph=True)[0],abar,a)
        return grad_output, None, gfa

InvLinOp = InvLinOp_Impl.apply
specfun = specfun_Impl.apply

def KernelLinearSolver(K,x,b,lmbda=0,eps=1e-6,precond=False,precondKernel=None):

    arraytype = type(b)

    if arraytype == np.ndarray:
        backend = np
        dtype = x.dtype.name
        copy = np.copy
        tile = np.tile
        solve = np.linalg.solve
        norm = np.linalg.norm
        Genred = GenredNumpy
        rand = lambda m, n : np.random.rand(m,n).astype(dtype)
        randn = lambda m, n : np.random.randn(m,n).astype(dtype)
        zeros = lambda shape : np.zeros(shape).astype(dtype)
        eye = lambda n : np.eye(n).astype(dtype)
        array = lambda x : np.array(x).astype(dtype)
        arraysum = np.sum
        transpose = lambda x : x.T
        numpy = lambda x : x
    elif arraytype == torch.Tensor:
        backend = torch
        torchdtype = x.dtype
        dtype = 'float32' if torchdtype==torch.float32 else 'float64'
        torchdeviceId = b.device
        KeOpsdeviceId = torchdeviceId.index  # id of Gpu device (in case Gpu is  used)
        copy = torch.clone
        tile = torch.Tensor.repeat
        solve = lambda A, b : torch.gesv(b,A)[0].contiguous()
        norm = torch.norm
        Genred = GenredTorch
        rand = lambda m, n : torch.rand(m,n, dtype=torchdtype, device=torchdeviceId)
        randn = lambda m, n : torch.randn(m,n, dtype=torchdtype, device=torchdeviceId)
        zeros = lambda shape : torch.zeros(shape, dtype=torchdtype, device=torchdeviceId)
        eye = lambda n : torch.eye(n, dtype=torchdtype, device=torchdeviceId)
        array = lambda x : torch.tensor(x, dtype=torchdtype, device=torchdeviceId)
        randn = lambda m, n : torch.randn(m,n, dtype=torchdtype, device=torchdeviceId)
        arraysum = lambda x, axis=None : x.sum() if axis is None else x.sum(dim=axis)
        transpose = lambda x : x.t
        numpy = lambda x : x.cpu().numpy()

    def PreconditionedConjugateGradientSolver(linop,b,invprecondop,eps=1e-6):
        # Preconditioned conjugate gradient algorithm to solve linear system of the form
        # Ma=b where linop is a linear operation corresponding
        # to a symmetric and positive definite matrix
        # invprecondop is linear operation corresponding to the inverse of the preconditioner matrix
        a = 0
        r = copy(b)
        z = invprecondop(r)
        p = copy(z)
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
        print("numiters=",k)
        return a

    def NystromInversePreconditioner(K,Kspec,x,lmbda):
        N,D = x.shape
        m = int(np.sqrt(N))
        ind = np.random.choice(range(N),m,replace=False)
        u = x[ind,:]
        start = time.time()
        M = K(u,u) + Kspec(tile(u,(m,1)),tile(u,(1,m)).reshape(-1,D),x).reshape(m,m)
        end = time.time()    
        print('Time for init:', round(end - start, 5), 's')
        def invprecondop(r):
            a = solve(M,K(u,x,r))
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
        my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=dtype)
        oos2 = array([1.0/sigma**2])
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
        my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=dtype)
        oos2 = array([1.0/sigma**2])
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
                sqdist += (x[:,k][:,None]-transpose(y[:,k][:,None]))**2
            return backend.exp(-oos2*sqdist)
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
        a = ConjugateGradientSolver(KernelLinOp,b,eps)
        
    return a


