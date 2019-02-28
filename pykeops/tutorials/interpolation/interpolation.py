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

from pykeops.tutorials.interpolation.torch.linsolve import InvKernelOp as InvKernelOp_pytorch
from pykeops.tutorials.interpolation.numpy.linsolve import InvKernelOp as InvKernelOp_numpy

from matplotlib import pyplot as plt

dtype = 'float64'  # May be 'float32' or 'float64'
useGpu = "auto"   # may be True, False or "auto"
backend = torch   # np or torch

if backend==np:
    Genred = GenredNumpy
    InvKernelOp = InvKernelOp_numpy
else:
    Genred = GenredTorch
    InvKernelOp = InvKernelOp_pytorch

# testing availability of Gpu: 
if not(useGpu==True):
    if backend == np:
        try:
            import GPUtil
            useGpu = len(GPUtil.getGPUs())>0
        except:
            useGpu = False
    else:
        useGpu = torch.cuda.is_available()

torchdtype = torch.float32 if dtype == 'float32' else torch.float64
torchdeviceId = torch.device('cuda:0') if useGpu else 'cpu'

 
def GaussKernel(D,Dv,sigma):
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                 'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
                 'b = Vy(' + str(Dv) + ')',  # Third arg  : j-variable, of size Dv
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=dtype)
    oos2 = array(np.array([1.0/sigma**2]).astype(dtype))
    def K(x,y,b):
        return my_routine(x,y,b,oos2)
    return K

def LinearSolver(K,x,b,lmbda=0):
    N = x.shape[0]
    M = K(x,x) + lmbda*eye(N)
    return solve(M,b)

def WarmUpGpu():
    # dummy first calls for accurate timing in case of GPU use
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    variables = ['x = Vx(1)',  # First arg   : i-variable, of size 1
                 'y = Vy(1)',  # Second arg  : j-variable, of size 1
                 'b = Vy(1)',  # Third arg  : j-variable, of size 1
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=dtype)
    dum = rand(10,1)
    dum2 = rand(10,1)
    my_routine(dum,dum,dum2,array([1.0]))
    my_routine(dum,dum,dum2,array([1.0]))
  
def numpy(x):
    if type(x) == np.ndarray:
        return x
    else:
        return x.cpu().numpy()

def array(x):
    if backend == np:
        return x
    else:
        return torch.tensor(x, dtype=torchdtype, device=torchdeviceId)

def transposeNumpy(x):
    return x.T
def transposeTorch(x):
    return torch.transpose(x,0,1)
def eyeNumpy(n):
    return np.eye(n).astype(dtype)
def eyeTorch(n):
    return torch.eye(n, dtype=torchdtype, device=torchdeviceId)
def solveNumpy(A,b):
    return np.linalg.solve(A,b)
def solveTorch(A,b):
    x, dum = torch.gesv(b,A)
    return x
if backend == np:
    transpose = transposeNumpy
    eye = eyeNumpy
    solve = solveNumpy
else:
    transpose = transposeTorch
    eye = eyeTorch
    solve = solveTorch
    
#######################################
#  We wrap this example into a function
#

def InterpolationExample(N,D,Dv,sigma,lmbda,eps=1e-6):
    print("")
    print('Interpolation example with ' + str(N) + ' points in ' + str(D) + '-D, sigma=' + str(sigma) + ', and lmbda=' + str(lmbda))

    #####################
    # Define our dataset
    #
    x = np.random.rand(N, D).astype(dtype)
    if D==1 & Dv==1:
        rx = np.reshape(np.sqrt(np.sum(x**2,axis=1)),[N,1])
        b = rx+.5*np.sin(6*rx)+.1*np.random.rand(N, 1).astype(dtype)
    else:
        b = np.random.randn(N, Dv).astype(dtype)
    #######################
    # Define the kernel
    #
    K = GaussKernel(D,Dv,sigma)
    
    ##########################
    # Perform the computations
    #       
    _x = array(x)
    _b = array(b)
    
    start = time.time()
    _a = KernelLinearSolver(K,_x,_b,lmbda,eps,precond=False)
    end = time.time()
    
    print('Time to perform (conjugate gradient solver):', round(end - start, 5), 's')
    print('L2 norm of the residual:', np.linalg.norm(numpy(K(_x,_x,_a)+lmbda*_a-_b)))
    
    start = time.time()
    _a = KernelLinearSolver(("gaussian",D,Dv,sigma),_x,_b,lmbda,eps,precond=True)
    end = time.time()
    
    print('Time to perform (preconditioned conjugate gradient solver):', round(end - start, 5), 's')
    print('L2 norm of the residual:', np.linalg.norm(numpy(K(_x,_x,_a)+lmbda*_a-_b)))
    
    start = time.time()
    _a = LinearSolver(GaussKernelMatrix(sigma),_x,_b,lmbda)
    end = time.time()
    
    print('Time to perform (usual matrix solver, no keops):', round(end - start, 5), 's')
    print('L2 norm of the residual:', np.linalg.norm(GaussKernelMatrix(sigma)(_x,_x)@_a+lmbda*_a-_b))
    
    if (D == 1):
        plt.ion()
        plt.clf()
        plt.scatter(x[:, 0], b[:, 0], s=10)
        t = np.reshape(np.linspace(0,1,1000),[1000,1]).astype(dtype)
        xt = K(array(t),_x,_a)
        plt.plot(t,numpy(xt),"r")
        print('Close the figure to continue.')
        plt.show(block=(__name__ == '__main__'))
 

eps = 1e-10
if useGpu:
    WarmUpGpu()
    InterpolationExample(N=10000,D=3,Dv=1,sigma=.1,lmbda=.1,eps=eps)   
else:
    InterpolationExample(N=1000,D=3,Dv=3,sigma=.1,lmbda=.1,eps=eps)
print("Done.")
