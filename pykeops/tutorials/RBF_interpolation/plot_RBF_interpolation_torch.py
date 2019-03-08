"""
=================
RBF Interpolation
=================

Example of Radial Basis Function interpolation
"""

#############################
#  Standard imports
#
import torch
import time
from pykeops.torch import Genred
from pykeops.torch.operations import InvKernelOp
from pykeops.torch.utils import WarmUpGpu
from matplotlib import pyplot as plt

useGpu = torch.cuda.is_available()
dtype = 'float32'
torchdtype = torch.float32 if dtype == 'float32' else torch.float64
torchdeviceId = torch.device('cuda:0') if useGpu else 'cpu'

#######################################
#  We wrap this example into a function
#

def InterpolationExample(N,D,Dv,sigma,lmbda):
    print("")
    print('Interpolation example with ' + str(N) + ' points in ' + str(D) + '-D, sigma=' + str(sigma) + ', and lmbda=' + str(lmbda))

    #####################
    # Define our dataset
    #
    x = torch.rand(N, D, dtype=torchdtype, device=torchdeviceId)
    if D==1 & Dv==1:
        rx = torch.reshape(torch.sqrt(torch.sum(x**2,dim=1)),[N,1])
        b = rx+.5*torch.sin(6*rx)+.1*torch.sin(20*rx)+.01*torch.randn(N, 1, dtype=torchdtype, device=torchdeviceId)
    else:
        b = torch.randn(N, Dv, dtype=torchdtype, device=torchdeviceId)
    oos2 = torch.tensor([1.0/sigma**2], dtype=torchdtype, device=torchdeviceId)

    # define the kernel : here a gaussian kernel
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    aliases = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                 'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
                 'b = Vy(' + str(Dv) + ')',  # Third arg  : j-variable, of size Dv
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
             
    # define the inverse kernel operation : here the 'b' argument specifies that linearity is with respect to variable b in formula.
    Kinv = InvKernelOp(formula, aliases, 'b', lmbda=lmbda, axis=1, cuda_type=dtype)
    
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
        plt.scatter(x.numpy()[:, 0], b.numpy()[:, 0], s=10)
        t = torch.reshape(torch.linspace(0,1,1000, dtype=torchdtype, device=torchdeviceId),[1000,1])
        K = Genred(formula, aliases, reduction_op='Sum', axis=1, cuda_type=dtype)
        xt = K(t,x,a,oos2)
        plt.plot(t.numpy(),xt.numpy(),"r")
        print('Close the figure to continue.')
        plt.show(block=(__name__ == '__main__'))
 
if useGpu:
    WarmUpGpu()
    InterpolationExample(N=10000,D=1,Dv=1,sigma=.1,lmbda=.1)   
else:
    InterpolationExample(N=1000,D=1,Dv=1,sigma=.1,lmbda=.1)
print("Done.")
