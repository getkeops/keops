"""
=======================================================================
K Nearest Neighbour algorithm using the generic syntax (NumPy bindings)
=======================================================================

We define a dataset of :math:`N` points in :math:`\mathbb R^D` with two classes, 
then apply a simple k-NN algorithm to evaluate the classifier over a grid.
This example uses a pure NumPy framework (without Pytorch).
See :ref:`here<sphx_glr__auto_tutorials_kmeans_plot_kmeans_pytorch.py>` for the equivalent script using PyTorch bindings.
"""

#############################
#  Standard imports
#

import time
import numpy as np
from pykeops.numpy import Genred
from pykeops.numpy.utils import IsGpuAvailable, WarmUpGpu

from matplotlib import pyplot as plt

type = 'float32'  # May be 'float32' or 'float64'

#######################################
#  We wrap this example into a function
#

def KNN_Example(N,D,Ktab):

    #####################
    # Define our dataset
    # cl gives class for each point in x
    #
    x = np.random.rand(N, D).astype(type)
    def fth(x):
        return 3*x*(x-.5)*(x-1)+x
    cl = x[:,1] < fth(x[:,0])
    x = x + .01*np.random.randn(N, D).astype(type)

    # building the grid for evaluation
    NG = 500
    tmp = np.linspace(0,1,NG).astype(type)
    g1, g2 = np.meshgrid(tmp,tmp)
    g = np.concatenate((g1.reshape(-1,1),g2.reshape(-1,1)),axis=1)
        
    plt.ion()
    plt.clf()    
    for i in range(len(Ktab)):
        K = Ktab[i]
        print("")
        print('k-NN example with ' + str(N) + ' points in ' + str(D) + '-D, and K=' + str(K))

        #######################
        # Define the kernel
        #
        formula = 'SqDist(x,y)'
        variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                     'y = Vy(' + str(D) + ')']  # Second arg  : j-variable, of size D

        # The parameter reduction_op='ArgKMin' together with axis=1 means that the reduction operation
        # is a sum over the second dimension j. Thence the results will be an i-variable.
        my_routine = Genred(formula, variables, reduction_op='ArgKMin', axis=1, cuda_type=type, opt_arg=K)
    
        ##########################
        # Perform the computations
        #
        WarmUpGpu()
        start = time.time()
        # calling keops to compute the K nearest neighbours
        indKNN = my_routine(g,x).astype(int)
        # classify grid points
        clg = np.mean(cl[indKNN],axis=1)>.5
        end = time.time()
        print('Time to perform k-NN classification:', round(end - start, 5), 's')

        plt.subplot(np.ceil(len(Ktab)/4),np.min([len(Ktab),4]),i+1)
        plt.scatter(x[:, 0], x[:, 1], c=cl, s=10)
        #plt.scatter(g[:, 0], g[:, 1], c=clg, s=2, alpha=.5)
        # reshaping grid classes for display as image
        clg = np.reshape(clg,(NG,NG))
        plt.imshow(clg, extent=(0,1,0,1), origin='lower', alpha=.5)
    print('Close the figure to continue.')
    plt.show(block=(__name__ == '__main__'))
    
###############################################################
# First experiment with 5000 points, dimension 2 and 50 classes
#

KNN_Example(N=1000,D=2,Ktab=(1,3,5,7,9,11,13,15))

####################################################################
# Second experiment with 500000 points, dimension 60 and 5000 classes
# (only when GPU is available)
#

if IsGpuAvailable():
    KNN_Example(N=500000,D=2,K=5)
print("Done.")
