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
from pykeops.numpy import Genred

from matplotlib import pyplot as plt

type = 'float64'  # May be 'float32' or 'float64'
useGPU = "auto"   # may be True, False or "auto"

# testing availability of Gpu: 
if (useGPU!="False"):
    try:
        import GPUtil
        useGpu = len(GPUtil.getGPUs())>0
    except:
        useGpu = False


def ConjugateGradientSolver(linop,b,eps=1e-6):
    # Conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    a = 0
    r = np.copy(b)
    p = np.copy(r)
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
    r = np.copy(b)
    z = invprecondop(r)
    p = np.copy(z)
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
    M = K(u,u) + Kspec(np.resize(u,(m*m,D)),np.repeat(u,m,axis=0),x).reshape(m,m)
    end = time.time()    
    print('Time for init:', round(end - start, 5), 's')
    def invprecondop(r):
        a = np.linalg.solve(M,K(u,x,r))
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
    oos2 = np.array([1.0/sigma**2]).astype(type)
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
    my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)
    oos2 = np.array([1.0/sigma**2]).astype(type)
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
            sqdist += (x[:,k][:,None]-y[:,k][:,None].T)**2
        return np.exp(-oos2*sqdist)
    return f

def NumpyLinsolve(K,x,b,lmbda=0):
    N = x.shape[0]
    K = K(x,x) + lmbda*np.eye(N)
    return np.linalg.solve(K,b)

def WarmUpGpu():
    # dummy first calls for accurate timing in case of GPU use
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    variables = ['x = Vx(1)',  # First arg   : i-variable, of size 1
                 'y = Vy(1)',  # Second arg  : j-variable, of size 1
                 'b = Vy(1)',  # Third arg  : j-variable, of size 1
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)
    dum = np.random.rand(10,1).astype(type)
    dum2 = np.random.rand(10,1).astype(type)
    my_routine(dum,dum,dum2,np.array([1.0]).astype(type))
    my_routine(dum,dum,dum2,np.array([1.0]).astype(type))
  
#######################################
#  We wrap this example into a function
#

def InterpolationExample(N,D,sigma,lmbda,eps=1e-6):
    print("")
    print('Interpolation example with ' + str(N) + ' points in ' + str(D) + '-D, sigma=' + str(sigma) + ', and lmbda=' + str(lmbda))

    #####################
    # Define our dataset
    #
    x = np.random.rand(N, D).astype(type)
    if D==1:
        rx = np.reshape(np.sqrt((x**2).sum(axis=1)),[N,1])
        bb = rx+.5*np.sin(6*rx)+.1*np.random.rand(N, 1).astype(type)
    else:
        bb = np.random.randn(N, D).astype(type)
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
    print('L2 norm of the residual:', np.linalg.norm(K(x,x,a)+lmbda*a-bb))
    
    start = time.time()
    a = KernelLinsolve(K,x,bb,lmbda,eps,precond=True,precondKernel=GaussKernelNystromPrecond(D,sigma))
    end = time.time()
    
    print('Time to perform (preconditioned conjugate gradient solver):', round(end - start, 5), 's')
    print('L2 norm of the residual:', np.linalg.norm(K(x,x,a)+lmbda*a-bb))
    
    start = time.time()
    a = NumpyLinsolve(K,x,bb,lmbda)
    end = time.time()
    
    print('Time to perform (numpy, no keops):', round(end - start, 5), 's')
    print('L2 norm of the residual:', np.linalg.norm(K(x,x,a)+lmbda*a-bb))
    
    if (D == 1):
        plt.ion()
        plt.clf()
        plt.scatter(x[:, 0], bb[:, 0], s=10)
        t = np.reshape(np.linspace(0,1,1000),[1000,1]).astype(type)
        xt = K(t,x,a)
        plt.plot(t,xt,"r")
        print('Close the figure to continue.')
        plt.show(block=(__name__ == '__main__'))
 

eps = 1e-10
if useGpu:
    WarmUpGpu()
    InterpolationExample(N=10000,D=3,sigma=.1,lmbda=.1,eps=eps)   
else:
    InterpolationExample(N=1000,D=1,sigma=.1,lmbda=.0001,eps=eps)
print("Done.")
