"""
Example of KeOps k-means algorithm using the generic syntax. 
==============================================================

We define a dataset of N points in R^D, then apply a simple k-means algorithm.
This example uses a pure numpy framework (without Pytorch).
"""

#--------------------------------------------------------------#
#                     Standard imports                         #
#--------------------------------------------------------------#
import time
import numpy as np
from matplotlib import pyplot as plt

#--------------------------------------------------------------#
#   Please use the "verbose" compilation mode for debugging    #
#--------------------------------------------------------------#
# import sys, os.path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

from pykeops.numpy import Genred

# import pykeops
# pykeops.verbose = False

#--------------------------------------------------------------#
#                   Define our dataset                         #
#--------------------------------------------------------------#
N = 5000
D = 2
K = 50
Niter = 10
print("k-means example with "+str(N)+" points in "+str(D)+"-D, and K="+str(K))

type = 'float32'  # May be 'float32' or 'float64'

x = np.random.rand(N,D).astype(type)

#--------------------------------------------------------------#
#                        Kernel                                #
#--------------------------------------------------------------#
formula = 'SqDist(x,y)'
variables = ['x = Vx('+str(D)+')',  # First arg   : i-variable, of size D
             'y = Vy('+str(D)+')']  # Second arg  : j-variable, of size D

# The parameter reduction_op='ArgMin' together with axis=1 means that the reduction operation
# is a sum over the second dimension j. Thence the results will be an i-variable.
my_routine = Genred(formula, variables, reduction_op='ArgMin', axis=1, cuda_type=type)

# dummy first call for accurate timing in case of GPU use
my_routine(np.random.rand(10,D).astype(type),np.random.rand(10,D).astype(type), backend="auto")

start = time.time()
# x is dataset, 
# c are centers, 
# cl is class index for each point in x
c = np.copy(x[:K,:])
for i in range(Niter):
    cl = my_routine(x,c,backend="auto").astype(int).reshape(N)
    c[:] = 0
    Ncl = np.bincount(cl).astype(type)
    for d in range(D):
        c[:,d] = np.bincount(cl,weights=x[:,d])
    c = (c.transpose()/Ncl).transpose()
end = time.time()
print("Time to perform",str(Niter),"iterations of k-means:",round(end-start,5),"s")
print("Time per iteration :",round((end-start)/Niter,5),"s")

if (D==2) and (N<100000):
    plt.ion()
    plt.clf()
    plt.scatter(x[:,0],x[:,1],c=cl,s=10)
    plt.scatter(c[:,0],c[:,1],c="black",s=50,alpha=.5)
    print("Done. Close the figure to exit.")
    plt.show(block=(__name__=="__main__"))
