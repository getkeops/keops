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
        #print(nr2new)
        if nr2new < eps**2:
            break
        p = r + (nr2new/nr2)*p
        nr2 = nr2new
        k += 1
    if eps==1e-6:
        print('k=',k)
    return a

def PreconditionedConjugateGradientSolver(linop,b,invprecondop,eps=1e-6):
    # Preconditioned conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    # invprecondop is linear operation corresponding to the inverse of the preconditioner matrix
    a = 0
    r = np.copy(b)
    z = invprecondop(r,eps)
    p = np.copy(z)
    rz = (r*z).sum()
    k = 0
    while True:    
        alpha = rz/(p*linop(p)).sum()
        a += alpha*p
        r -= alpha*linop(p)
        if (r**2).sum() < eps**2:
            break
        z = invprecondop(r,eps)
        rznew = (r*z).sum()
        p = z + (rznew/rz)*p
        rz = rznew
        k += 1
    print('k=',k)
    return a

#######################################
#  We wrap this example into a function
#

def InterpolationExample(N,D,sigma,lmbda):
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
    formula = 'Inv(oos2)*Exp(-IntInv(2)*oos2*SqDist(x,y))*b'
    variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                 'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
                 'b = Vy(' + str(1) + ')',  # Third arg  : j-variable, of size 1
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter

    # The parameter reduction_op='Sum' together with axis=1 means that the reduction operation
    # is a sum over the second dimension j. Thence the results will be an i-variable.
    my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)
    
    def K(x,y,b):
        return my_routine(x,y,b,np.array([1.0/sigma**2]).astype(type))
        
    def KernelLinOp(a):
        return K(x,x,a) + lmbda*a

    ind = np.random.choice(range(N),int(np.sqrt(N)))
    u = x[ind,:]
    def f(a):
        return lmbda*K(u,u,a)+K(u,x,K(x,u,a))
    def NystromInversePreconditioner(r,eps=1e-8):
        ru = K(u,x,r)
        ru = ConjugateGradientSolver(f,ru,eps)
        z = (r - K(x,u,ru))/lmbda
        return z
               
    ##########################
    # Perform the computations
    #
    
    # dummy first calls for accurate timing in case of GPU use
    dum = np.random.rand(10,D).astype(type)
    dum2 = np.random.rand(10,1).astype(type)
    my_routine(dum,dum,dum2,np.array([1.0]).astype(type))
    my_routine(dum,dum,dum2,np.array([1.0]).astype(type))
    
    start = time.time()
    a = ConjugateGradientSolver(KernelLinOp,b)
    end = time.time()
    
    print('Time to perform:', round(end - start, 5), 's')
    print('accuracy:', np.linalg.norm(KernelLinOp(a)-b))
    
#    start = time.time()
#    a = PreconditionedConjugateGradientSolver(KernelLinOp,b,NystromInversePreconditioner)
#    end = time.time()
    
#    print('Time to perform:', round(end - start, 5), 's')
#    print('accuracy:', np.linalg.norm(KernelLinOp(a)-b))
    
    if (D == 1):
        plt.ion()
        plt.clf()
        plt.scatter(x[:, 0], b[:, 0], s=10)
        t = np.reshape(np.linspace(0,1,1000),[1000,1])
        xt = K(t,x,a)
        plt.plot(t,xt,"r")
        print('Close the figure to continue.')
        plt.show(block=(__name__ == '__main__'))
 
################################################
# First experiment with 5000 points, dimension 2
#

InterpolationExample(N=500,D=1,sigma=.1,lmbda=.1)

####################################################################
# Second experiment with 500000 points, dimension 60 and 5000 classes
# (only when GPU is available)
#

#import GPUtil
#if len(GPUtil.getGPUs())>0:
#    KMeansExample(N=500000,D=60,sigma=.1,lmbda=1.0)
print("Done.")
