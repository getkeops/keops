"""
=============
Interpolation
=============

Example of interpolation
"""

#############################
#  Standard imports
#

import time
import numpy as np
import torch
from pykeops.numpy import Genred as GenredNumpy
from pykeops.torch import Genred as GenredTorch

from matplotlib import pyplot as plt

type = 'float64'  # May be 'float32' or 'float64'
useGpu = "auto"   # may be True, False or "auto"
backend = torch # np or torch

# testing availability of Gpu: 
if not(useGpu):
    if backend == np:
        try:
            import GPUtil
            useGpu = len(GPUtil.getGPUs())>0
        except:
            useGpu = False
    else:
        useGpu = torch.cuda.is_available()

def solveNumpy(A,b):
    return np.linalg.solve(A,b)
def solveTorch(A,b):
    x, dum = torch.gesv(b,A)
    return x.contiguous()
def randNumpy(m,n):
    return np.random.rand(m,n).astype(type)
def randTorch(m,n):
    return torch.rand(m,n, dtype=torchdtype, device=torchdeviceId)
def randnNumpy(m,n):
    return np.random.randn(m,n).astype(type)
def randnTorch(m,n):
    return torch.randn(m,n, dtype=torchdtype, device=torchdeviceId)
def zerosNumpy(shape):
    return np.zeros(shape).astype(type)
def zerosTorch(shape):
    return torch.zeros(shape, dtype=torchdtype, device=torchdeviceId)
def eyeNumpy(n):
    return np.eye(n).astype(type)
def eyeTorch(n):
    return torch.eye(n, dtype=torchdtype, device=torchdeviceId)
def arrayNumpy(x):
    return np.array(x).astype(type)
def arrayTorch(x):
    return torch.tensor(x, dtype=torchdtype, device=torchdeviceId)
def sumNumpy(x,axis):
    return x.sum(axis=axis)
def sumTorch(x,axis=None):
    if axis is None:
        return x.sum()
    else:
        return x.sum(dim=axis)
def transposeNumpy(x):
    return x.T
def transposeTorch(x):
    return torch.transpose(x,0,1)
def numpyNumpy(x):
    return x
def numpyTorch(x):
    return x.cpu().numpy()

def SetBackendFunctions():
    global solve, copy, tile, solve, norm, Genred, rand, eye, zeros, array, randn, arraysum, transpose, numpy
    if backend == np:
        copy = np.copy
        tile = np.tile
        solve = solveNumpy
        norm = np.linalg.norm
        Genred = GenredNumpy
        rand = randNumpy
        zeros = zerosNumpy
        eye = eyeNumpy
        array = arrayNumpy
        randn = randnNumpy
        arraysum = sumNumpy
        transpose = transposeNumpy
        numpy = numpyNumpy
    else:
        copy = torch.clone
        tile = torch.Tensor.repeat
        solve = solveTorch
        norm = torch.norm
        Genred = GenredTorch
        rand = randTorch
        zeros = zerosTorch
        eye = eyeTorch
        array = arrayTorch
        randn = randnTorch
        arraysum = sumTorch
        global torchdtype, torchdeviceId, KeOpsdeviceId, KeOpsdtype
        torchdtype = torch.float32 if type == 'float32' else torch.float64
        torchdeviceId = torch.device('cuda:0') if useGpu else 'cpu'
        KeOpsdeviceId = torchdeviceId.index  # id of Gpu device (in case Gpu is  used)
        KeOpsdtype = torchdtype.__str__().split('.')[1]  # 'float32'
        transpose = transposeTorch
        numpy = numpyTorch

SetBackendFunctions()






def ConjugateGradientSolver(linop,b,eps=1e-6):
    # Conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    a = 0
    r = copy(b)
    p = copy(r)
    nr2 = (r**2).sum()
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
    
def KernelLinsolve(K,x,b,lmbda=0,eps=1e-6,precond=False,precondKernel=None):
    def KernelLinOp(a):
        return K(x,x,a) + lmbda*a
    if precond:
        invprecondop = NystromInversePreconditioner(K,precondKernel,x,lmbda)
        a = PreconditionedConjugateGradientSolver(KernelLinOp,b,invprecondop,eps)
    else:
        a = ConjugateGradientSolver(KernelLinOp,b,eps)
    return a

def GaussKernel(D,sigma):
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                 'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
                 'b = Vy(' + str(D) + ')',  # Third arg  : j-variable, of size D
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)
    oos2 = array([1.0/sigma**2])
    KernelMatrix = GaussKernelMatrix(sigma)
    def K(x,y,b=None):
        if b is None:
            return KernelMatrix(x,y)
        else:
            return my_routine(x,y,b,oos2)
    return K

def GaussKernel_alt(D,sigma):
    formula = 'Exp(oos2*(x,y))*b'
    variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                 'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
                 'b = Vy(' + str(D) + ')',  # Third arg  : j-variable, of size D
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)
    oos2 = array([1.0/sigma**2])
    KernelMatrix = GaussKernelMatrix(sigma)
    def K(x,y,b=None):
        if b is None:
            return KernelMatrix(x,y)
        else:
            sqx = backend.exp(-oos2*(x**2).sum(axis=1))[:,np.newaxis]
            sqy = backend.exp(-oos2*(y**2).sum(axis=1))[:,np.newaxis]
            return sqx*my_routine(x,y,sqy*b,2*oos2)
    return K

def GaussKernelNystromPrecond(D,sigma):
    formula = 'Exp(-oos2*(SqDist(u,x)+SqDist(v,x)))'
    variables = ['u = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                 'v = Vx(' + str(D) + ')',  # Second arg  : i-variable, of size D
                 'x = Vy(' + str(D) + ')',  # Third arg  : j-variable, of size D
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)
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

def Linsolve(K,x,b,lmbda=0):
    N = x.shape[0]
    K = K(x,x) + lmbda*eye(N)
    return solve(K,b)

def WarmUpGpu():
    # dummy first calls for accurate timing in case of GPU use
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    variables = ['x = Vx(1)',  # First arg   : i-variable, of size 1
                 'y = Vy(1)',  # Second arg  : j-variable, of size 1
                 'b = Vy(1)',  # Third arg  : j-variable, of size 1
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)
    dum = rand(10,1)
    dum2 = rand(10,1)
    my_routine(dum,dum,dum2,array([1.0]))
    my_routine(dum,dum,dum2,array([1.0]))
  
#######################################
#  We wrap this example into a function
#

def InterpolationExample(N,D,sigma,lmbda,eps=1e-6):
    print("")
    print('Interpolation example with ' + str(N) + ' points in ' + str(D) + '-D, sigma=' + str(sigma) + ', and lmbda=' + str(lmbda))

    #####################
    # Define our dataset
    #
    x = rand(N, D)
    x[:,1:] = 0
    if D==1:
        rx = backend.reshape(backend.sqrt(arraysum(x**2,axis=1)),[N,1])
        bb = rx+.5*backend.sin(6*rx)+.1*rand(N, 1)
    else:
        bb = randn(N, D)
    #######################
    # Define the kernel
    #
    K = GaussKernel(D,sigma)
    
    ##########################
    # Perform the computations
    #       
    start = time.time()
    a = KernelLinsolve(K,x,bb,lmbda,eps,precond=False)
    end = time.time()
    
    print('Time to perform (conjugate gradient solver):', round(end - start, 5), 's')
    print('L2 norm of the residual:', norm(K(x,x,a)+lmbda*a-bb))
    
    start = time.time()
    a = KernelLinsolve(K,x,bb,lmbda,eps,precond=True,precondKernel=GaussKernelNystromPrecond(D,sigma))
    end = time.time()
    
    print('Time to perform (preconditioned conjugate gradient solver):', round(end - start, 5), 's')
    print('L2 norm of the residual:', np.linalg.norm(K(x,x,a)+lmbda*a-bb))
    
    start = time.time()
    a = Linsolve(K,x,bb,lmbda)
    end = time.time()
    
    print('Time to perform (usual matrix solver, no keops):', round(end - start, 5), 's')
    print('L2 norm of the residual:', norm(K(x,x,a)+lmbda*a-bb))
    
    if (D == 1):
        plt.ion()
        plt.clf()
        plt.scatter(x[:, 0], bb[:, 0], s=10)
        t = np.reshape(np.linspace(0,1,1000),[1000,1]).astype(type)
        xt = K(array(t),x,a)
        plt.plot(t,numpy(xt),"r")
        print('Close the figure to continue.')
        plt.show(block=(__name__ == '__main__'))
 

eps = 1e-10
if useGpu:
    WarmUpGpu()
    InterpolationExample(N=10000,D=3,sigma=.1,lmbda=.1,eps=eps)   
else:
    InterpolationExample(N=1000,D=1,sigma=.1,lmbda=.0001,eps=eps)
print("Done.")
