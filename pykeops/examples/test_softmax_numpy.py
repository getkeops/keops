"""
Example of KeOps softmax reduction using the generic syntax. 
The following operation is implemented :
    imputs : x array of size Mx3 representing M vectors in R^3
             y array of size Nx3 representing N vectors in R^3
             b array of size Nx2 representing N vectors in R^2
    output : z array of size Mx2 representing M vectors in R^2
             where z_i = sum_j exp(K(x_i,y_j))b_j / sum_j exp(K(x_i,y_j))
             with K(x_i,y_j) = |x_i-y_j|^2
This example uses the Numpy bindings
==================================================================
"""

#--------------------------------------------------------------#
#                     Standard imports                         #
#--------------------------------------------------------------#
import time
import numpy as np
from pykeops.numpy.operations import softmax

#--------------------------------------------------------------#
#                   Define our dataset                         #
#--------------------------------------------------------------#
M = 500
N = 400
D = 3
Dv = 2

x = 2*np.random.randn(M,D)
y = 2*np.random.randn(N,D)
b = np.random.rand(N,Dv)

#--------------------------------------------------------------#
#                        Kernel                                #
#--------------------------------------------------------------#
formula = 'SqDist(x,y)'
formula_weights = 'b'
variables = ['x = Vx('+str(D)+')',  # First arg   : i-variable, of size D
             'y = Vy('+str(D)+')',  # Second arg  : j-variable, of size D
             'b = Vy('+str(Dv)+')'] # third arg : j-variable, of size Dv

softmax_op = softmax(formula,formula_weights,variables)

start = time.time()
c = softmax_op(x, y, b)
print("Time to compute the softmax operation (KeOps implementation): ",round(time.time()-start,5),"s")

# compare with direct implementation
start = time.time()
cc = 0
for k in range(D):
    xk = x[:,k][:,None]
    yk = y[:,k][:,None]
    cc += (xk-yk.T)**2
cc -= np.max(cc,axis=1)[:,None] # subtract the max for robustness
cc = np.exp(cc)@b/np.sum(np.exp(cc),axis=1)[:,None]
print("Time to compute the softmax operation (direct implementation): ",round(time.time()-start,5),"s")

print("relative error : ", (np.linalg.norm(c-cc)/np.linalg.norm(c)).item())

