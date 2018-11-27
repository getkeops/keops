"""
=============
Interpolation
=============

Example of a interpolation
"""

#############################
#  Standard imports
#

import time
import numpy as np
from pykeops.numpy import Genred

from matplotlib import pyplot as plt

type = 'float32'  # May be 'float32' or 'float64'

#######################################
#  We wrap this example into a function
#

def InterpolationExample(N,D,sigma,lambda,Niter=10):
    print("")
    print('Interpolation example with ' + str(N) + ' points in ' + str(D) + '-D, sigma=' + str(sigma) + ', and lambda=' + str(lambda))

    #####################
    # Define our dataset
    #
    x = np.random.rand(N, D).astype(type)

    #######################
    # Define the kernel
    #
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                 'y = Vy(' + str(D) + ')']  # Second arg  : j-variable, of size D
                 'b = Vy(' + str(D) + ')']  # Third arg  : j-variable, of size D
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter

    # The parameter reduction_op='Sum' together with axis=1 means that the reduction operation
    # is a sum over the second dimension j. Thence the results will be an i-variable.
    my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=type)
    
    ##########################
    # Perform the computations
    #
    
    # dummy first calls for accurate timing in case of GPU use
    dum = np.random.rand(10,D).astype(type)
    my_routine(dum,dum)
    my_routine(dum,dum)
    
    start = time.time()
    # x is dataset
    # we want to solve Ma=b with M=K(x)+lambda*I
    r = b - my_routine(x,x,b,1/sigma**2) - lambda*b
    p = r
    for k in range(Niter):
        Mp = my_routine(x,x,p,1/sigma**2)
        normr2 = (r**2).sum()
        alpha = normr2/(p,Mp).sum()
        x += alpha*p
        r += alpha*Mp
        beta = (r**2).sum()/normr2
        p = r + beta*p
    
    end = time.time()
    print('Time to perform', str(Niter), 'iterations of k-means:', round(end - start, 5), 's')
    print('Time per iteration :', round((end - start) / Niter, 5), 's')
    
    if (D == 2):
        plt.ion()
        plt.clf()
        plt.scatter(x[:, 0], x[:, 1], c=cl, s=10)
        print('Close the figure to continue.')
        plt.show(block=(__name__ == '__main__'))
 
###############################################################
# First experiment with 5000 points, dimension 2 and 50 classes
#

KMeansExample(N=5000,D=2,K=50)

####################################################################
# Second experiment with 500000 points, dimension 60 and 5000 classes
# (only when GPU is available)
#

import GPUtil
if len(GPUtil.getGPUs())>0:
    KMeansExample(N=500000,D=60,K=5000)
print("Done.")
