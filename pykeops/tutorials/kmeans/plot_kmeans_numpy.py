"""
===========================================================
K-means algorithm using the generic syntax (NumPy bindings)
===========================================================

We define a dataset of :math:`N` points in :math:`\mathbb R^D`, then apply a simple k-means algorithm.
This example uses a pure NumPy framework (without Pytorch).
See :ref:`here<sphx_glr__auto_tutorials_kmeans_plot_kmeans_pytorch.py>` for the equivalent script using PyTorch bindings.
"""

#############################
#  Standard imports
#

import time
import numpy as np
from pykeops.numpy import Genred
from pykeops.numpy.utils import IsGpuAvailable

from matplotlib import pyplot as plt

type = 'float32'  # May be 'float32' or 'float64'

#######################################
#  We wrap this example into a function
#

def KMeansExample(N,D,K,Niter=10):
    print("")
    print('k-means example with ' + str(N) + ' points in ' + str(D) + '-D, and K=' + str(K))

    #####################
    # Define our dataset
    #
    x = np.random.rand(N, D).astype(type)

    #######################
    # Define the kernel
    #
    formula = 'SqDist(x,y)'
    variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                 'y = Vy(' + str(D) + ')']  # Second arg  : j-variable, of size D

    # The parameter reduction_op='ArgMin' together with axis=1 means that the reduction operation
    # is a sum over the second dimension j. Thence the results will be an i-variable.
    my_routine = Genred(formula, variables, reduction_op='ArgMin', axis=1, cuda_type=type)
    
    ##########################
    # Perform the computations
    #
    
    # dummy first calls for accurate timing in case of GPU use
    dum = np.random.rand(10,D).astype(type)
    my_routine(dum,dum)
    my_routine(dum,dum)
    
    start = time.time()
    # x is dataset, 
    # c are centers, 
    # cl is class index for each point in x
    c = np.copy(x[:K, :])
    for i in range(Niter):
        cl = my_routine(x,c).astype(int).reshape(N)
        c[:] = 0
        Ncl = np.bincount(cl).astype(type)
        for d in range(D):
            c[:, d] = np.bincount(cl, weights=x[:, d])
        c = (c.transpose() / Ncl).transpose()
    end = time.time()
    print('Time to perform', str(Niter), 'iterations of k-means:', round(end - start, 5), 's')
    print('Time per iteration :', round((end - start) / Niter, 5), 's')
    
    if (D == 2):
        plt.ion()
        plt.clf()
        plt.scatter(x[:, 0], x[:, 1], c=cl, s=10)
        plt.scatter(c[:, 0], c[:, 1], c='black', s=50, alpha=.5)
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

if IsGpuAvailable():
    KMeansExample(N=500000,D=60,K=5000)
print("Done.")
