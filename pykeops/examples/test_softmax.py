"""
Example of KeOps softmax reduction using the generic syntax. 
The following operation is implemented :
    imputs : x array of size Mx3 representing M vectors in R^3
             y array of size Nx3 representing N vectors in R^3
             b array of size Nx2 representing N vectors in R^2
    output : z array of size Mx2 representing M vectors in R^2
             where z_i = sum_j exp(K(x_i,y_j))b_j / sum_j exp(K(x_i,y_j))
             with K(x_i,y_j) = |x_i-y_j|^2
==================================================================
"""

#--------------------------------------------------------------#
#                     Standard imports                         #
#--------------------------------------------------------------#
import time

backend  = 'torch'  # May be 'numpy' or 'torch'
dtype = 'float32'  # May be 'float32' or 'float64'
use_cuda = False

if backend=='numpy':
    import numpy as np
    from pykeops.numpy import Genred
    rand = lambda m,n : np.random.rand(m,n).astype(dtype)
    randn = lambda m,n : np.random.randn(m,n).astype(dtype)
    exp = np.exp
    arraysum = lambda x, axis=0 : np.sum(x,axis=axis)
    transpose = lambda x : x.T
    norm = np.linalg.norm
    arraymax = lambda x, axis=0 : np.max(x,axis=axis)
elif backend=='torch':
    import torch
    from pykeops.torch import Genred
    torchdtype = torch.float32 if dtype == 'float32' else torch.float64
    use_cuda = torch.cuda.is_available() if use_cuda else False
    device = 'cuda' if use_cuda else 'cpu'
    rand = lambda m,n : torch.rand(m,n,dtype=torchdtype,device=device)
    randn = lambda m,n : torch.randn(m,n,dtype=torchdtype,device=device)
    exp = torch.exp
    arraysum = lambda x, axis=0 : torch.sum(x,dim=axis)
    transpose = lambda x : x.t()
    norm = torch.norm
    arraymax = lambda x, axis=0 : torch.max(x,dim=axis)[0]

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

if use_cuda:
    WarmUpGpu()
    
#--------------------------------------------------------------#
#                   Define our dataset                         #
#--------------------------------------------------------------#
M = 500
N = 400
D = 3
Dv = 2

x = 2*randn(M,D)
y = 2*randn(N,D)
b = rand(N,Dv)

#--------------------------------------------------------------#
#         Custom function for the softmax operation            #
#--------------------------------------------------------------#

def softmax(formula,formula_weights,variables):
    formula2 = 'Concat(IntCst(1),' + formula_weights + ')'
    my_routine = Genred(formula, variables, reduction_op='LogSumExpVect', axis=1, cuda_type=dtype, formula2=formula2)
    def f(*args):
        out = my_routine(*args, backend="auto")
        out = out[:,2:]/out[:,1][:,None]
        return out
    return f

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
    cc += (xk-transpose(yk))**2
cc -= arraymax(cc,axis=1)[:,None] # subtract the max for robustness
cc = exp(cc)@b/arraysum(exp(cc),axis=1)[:,None]
print("Time to compute the softmax operation (direct implementation): ",round(time.time()-start,5),"s")

print("relative error : ", (norm(c-cc)/norm(c)).item())

