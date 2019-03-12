"""
=============================
K-NN classification (pytorch)
=============================

We define a dataset of :math:`N` points in :math:`\mathbb R^D` with two classes, 
then apply a simple k-Nearest Neighbour algorithm to evaluate the classifier over a grid.
This example uses the PyTorch framework.
See :ref:`here<sphx_glr__auto_tutorials_KNN_plot_knn_numpy.py>` for the equivalent script using NumPy bindings.
"""

#############################
#  Standard imports
#

import time
import numpy as np
import torch
from pykeops.torch import Genred
from pykeops.torch.utils import WarmUpGpu

from matplotlib import pyplot as plt

dtype = 'float32'  # May be 'float32' or 'float64'
torchtype = torch.float32 if dtype == 'float32' else torch.float64

if torch.cuda.is_available():
    torchdeviceId = torch.device('cuda:0')
    KeOpsdeviceId = torchdeviceId.index
else:
    torchdeviceId = torch.device('cpu')
    
#######################################
#  We wrap this example into a function
#

def KNN_Example(N,D,Ng,Ktab):

    #####################
    # Define our dataset
    # cl gives class for each point in x
    #
    x = torch.rand(N, D,dtype=torchtype,device=torchdeviceId)
    def fth(x):
        return 3*x*(x-.5)*(x-1)+x
    cl = (x[:,1]+.03*torch.randn(N,dtype=torchtype,device=torchdeviceId)) < fth(x[:,0])

    # building the grid for evaluation
    tmp = torch.linspace(0,1,Ng,dtype=torchtype,device=torchdeviceId)
    g1, g2 = torch.meshgrid(tmp,tmp)
    g1, g2 = g1.t(), g2.t()
    g = torch.cat((g1.reshape(-1,1),g2.reshape(-1,1)),dim=1)

    WarmUpGpu()
        
    if D==2:
        plt.ion()
        plt.clf()  
        plt.subplot(np.ceil((len(Ktab)+1)/4),np.min([(len(Ktab)+1),4]),1)  
        plt.scatter(x.cpu().numpy()[:, 0], x.cpu().numpy()[:, 1], c=cl.cpu().numpy(), s=2)
        plt.imshow(np.ones((2,2)), extent=(0,1,0,1), alpha=0)
        plt.axis('off')
        plt.title('data points')
    for i in range(len(Ktab)):
        K = Ktab[i]
        print("")
        print('K-NN example with ' + str(N) + ' points in ' + str(D) + '-D, and K=' + str(K))

        #######################
        # Define the kernel
        #
        formula = 'SqDist(x,y)'
        variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                     'y = Vy(' + str(D) + ')']  # Second arg  : j-variable, of size D

        # The parameter reduction_op='ArgKMin' together with axis=1 means that the reduction operation
        # is a sum over the second dimension j. Thence the results will be an i-variable.
        my_routine = Genred(formula, variables, reduction_op='ArgKMin', axis=1, cuda_type=dtype, opt_arg=K)
    
        ##########################
        # Perform the computations
        #
        start = time.time()
        # calling keops to compute the K nearest neighbours
        indKNN = my_routine(g,x).long()
        # classify grid points
        clg = torch.mean(cl.float()[indKNN],dim=1)>.5
        end = time.time()
        print('Time to perform K-NN classification over',Ng*Ng,'test points:', round(end - start, 5), 's')

        if D==2:
            plt.subplot(np.ceil((len(Ktab)+1)/4),np.min([(len(Ktab)+1),4]),i+2)
            # reshaping grid classes for display as image
            clg = np.reshape(clg.cpu().numpy(),(Ng,Ng))
            plt.imshow(clg, extent=(0,1,0,1), origin='lower')
            plt.axis('off')
            plt.title(str(K)+'-NN classifier')
    if D==2:
        print('Close the figure to continue.')
        plt.show(block=(__name__ == '__main__'))
    
if torch.cuda.is_available():
    ######################################################
    # On Gpu : experiment with 10000 points in dimension 2
    # with evalaution over a grid of size 2000*2000
    #
    KNN_Example(N=10000,D=2,Ng=2000,Ktab=(1,3,5,9,15,21,35))
else:
    #######################################################
    # Cpu only : experiment with 1000 points in dimension 2
    # with evalaution over a grid of size 500*500
    #
    KNN_Example(N=1000,D=2,Ng=500,Ktab=(1,3,5,9,15,21,35))

print("Done.")
