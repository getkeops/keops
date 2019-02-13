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
from pykeops.torch import Genred

from matplotlib import pyplot as plt

type = 'float64'  # May be 'float32' or 'float64'
torchtype = torch.float32 if type == 'float32' else torch.float64

useGPU = "auto"   # may be True, False or "auto"

# testing availability of Gpu: 
if (useGPU!="False"):
    useGpu = torch.cuda.is_available()

def ConjugateGradientSolver(linop,b,eps=1e-6):
    # Conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    a = 0
    r = b.clone()
    p = r.clone()
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
    return a

def PreconditionedConjugateGradientSolver(linop,b,invprecondop,eps=1e-6):
    # Preconditioned conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    # invprecondop is linear operation corresponding to the inverse of the preconditioner matrix
    a = 0
    r = b.clone()
    z = invprecondop(r)
    p = z.clone()
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

def Kmeans(x,K,Niter=2):
    N,D = x.shape
    formula = 'SqDist(x,y)'
    variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                 'y = Vy(' + str(D) + ')']  # Second arg  : j-variable, of size D
    my_routine = Genred(formula, variables, reduction_op='ArgMin', axis=1, cuda_type=type)
    c = x[:K, :].clone()
    for i in range(Niter):
        cl = my_routine(x,c).astype(int).reshape(N)
        c[:] = 0
        Ncl = torch.bincount(cl).float()
        for d in range(D):
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl
    return c

def NystromInversePreconditioner(K,x,lmbda):
    N,D = x.shape
    m = int(np.sqrt(N))

    ind = np.random.choice(range(N),m,replace=False)
    u = x[ind,:]
    
    #u = Kmeans(x,m)
    
    start = time.time()
    M = torch.zeros(m,m,dtype=torchtype,device=torchdeviceId)
    one = torch.ones(1,1,dtype=torchtype,device=torchdeviceId)
    for j in range(m):
        Kuuj = K(u,u[j,:].reshape((1,D)),one)
        Kxuj = K(x,u[j,:].reshape((1,D)),one)
        M[:,j] = (Kuuj + K(u,x,Kxuj)).flatten()
    end = time.time()    
    print('Time for init:', round(end - start, 5), 's')

    def invprecondop(r):
        return (r - K(x,u,np.linalg.solve(M,K(u,x,r))))/lmbda
    return invprecondop
    
def KernelLinsolve(K,x,b,lmbda=0,eps=1e-6,precond=False):
    def KernelLinOp(a):
        return K(x,x,a) + lmbda*a
    if precond:
        invprecondop = NystromInversePreconditioner(K,x,lmbda)
        a = PreconditionedConjugateGradientSolver(KernelLinOp,b,invprecondop,eps)
    else:
        a = ConjugateGradientSolver(KernelLinOp,b,eps)
    return a

def GaussKernel(D,sigma):
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                 'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
                 'b = Vy(' + str(1) + ')',  # Third arg  : j-variable, of size 1
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)
    oos2 = np.array([1.0/sigma**2]).astype(type)
    def K(x,y,b):
        return my_routine(x,y,b,oos2)
    return K

def WarmUpGpu():
    # dummy first calls for accurate timing in case of GPU use
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    variables = ['x = Vx(' + str(1) + ')',  # First arg   : i-variable, of size D
                 'y = Vy(' + str(1) + ')',  # Second arg  : j-variable, of size D
                 'b = Vy(' + str(1) + ')',  # Third arg  : j-variable, of size 1
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
    rx = np.reshape(np.sqrt((x**2).sum(axis=1)),[N,1])
    b = rx+.5*np.sin(6*rx)+.1*np.random.rand(N, 1).astype(type)

    #######################
    # Define the kernel
    #
    K = GaussKernel(D,sigma)
    
    ##########################
    # Perform the computations
    #       
    start = time.time()
    a = KernelLinsolve(K,x,b,lmbda,eps,precond=False)
    end = time.time()
    
    print('Time to perform (conjugate gradient solver):', round(end - start, 5), 's')
    print('L2 norm of the residual:', np.linalg.norm(K(x,x,a)+lmbda*a-b))
    
    start = time.time()
    a = KernelLinsolve(K,x,b,lmbda,eps,precond=True)
    end = time.time()
    
    print('Time to perform (preconditioned conjugate gradient solver):', round(end - start, 5), 's')
    print('L2 norm of the residual:', np.linalg.norm(K(x,x,a)+lmbda*a-b))
    
    if (D == 1):
        plt.ion()
        plt.clf()
        plt.scatter(x[:, 0], b[:, 0], s=10)
        t = np.reshape(np.linspace(0,1,1000),[1000,1]).astype(type)
        xt = K(t,x,a)
        plt.plot(t,xt,"r")
        print('Close the figure to continue.')
        plt.show(block=(__name__ == '__main__'))
 

eps = 1e-10
if useGpu:
    WarmUpGpu()
    InterpolationExample(N=10000,D=1,sigma=.1,lmbda=.0001,eps=eps)   
else:
    InterpolationExample(N=1000,D=1,sigma=.1,lmbda=.1,eps=eps)
print("Done.")
