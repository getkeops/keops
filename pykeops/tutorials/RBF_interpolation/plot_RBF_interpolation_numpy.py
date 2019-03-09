"""
=============
Interpolation
=============

Example of interpolation
"""

#############################
#  Standard imports
#
import numpy as np
import time
from pykeops.numpy import Genred
from pykeops.numpy.operations import InvKernelOp
from pykeops.numpy.utils import IsGpuAvailable, WarmUpGpu
from matplotlib import pyplot as plt

useGpu = IsGpuAvailable()
dtype = 'float64'

#######################################
#  We wrap this example into a function
#

def InterpolationExample(N,D,Dv,sigma,lmbda):
    print("")
    print('Interpolation example with ' + str(N) + ' points in ' + str(D) + '-D, sigma=' + str(sigma) + ', and lmbda=' + str(lmbda))

    #####################
    # Define our dataset
    #
    x = np.random.rand(N, D).astype(dtype)
    if D==1 & Dv==1:
        rx = np.reshape(np.sqrt(np.sum(x**2,axis=1)),[N,1])
        b = rx+.5*np.sin(6*rx)+.1*np.sin(20*rx)+.01*np.random.randn(N, 1).astype(dtype)
    else:
        b = np.random.randn(N, Dv).astype(dtype)
    oos2 = np.array([1.0/sigma**2]).astype(dtype)

    # define the kernel : here a gaussian kernel
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    aliases = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                 'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
                 'b = Vy(' + str(Dv) + ')',  # Third arg  : j-variable, of size Dv
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
             
    # define the inverse kernel operation : here the 'b' argument specifies that linearity is with respect to variable b in formula.
    Kinv = InvKernelOp(formula, aliases, 'b', lmbda=lmbda, axis=1)
    
    ##########################
    # Perform the computations
    #       
    start = time.time()
    a = Kinv(x,x,b,oos2)
    end = time.time()
    
    print('Time to perform:', round(end - start, 5), 's')
    
    if (D == 1):
        plt.ion()
        plt.clf()
        plt.scatter(x[:, 0], b[:, 0], s=10)
        t = np.reshape(np.linspace(0,1,1000),[1000,1]).astype(dtype)
        K = Genred(formula, aliases, reduction_op='Sum', axis=1, cuda_type=dtype)
        xt = K(t,x,a,oos2)
        plt.plot(t,xt,"r")
        print('Close the figure to continue.')
        plt.show(block=(__name__ == '__main__'))
 
if useGpu:
    WarmUpGpu()
    InterpolationExample(N=10000,D=1,Dv=1,sigma=.1,lmbda=.1)   
else:
    InterpolationExample(N=1000,D=1,Dv=1,sigma=.1,lmbda=.1)
print("Done.")
