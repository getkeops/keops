

import numpy as np
from pykeops.numpy import Genred 

from pykeops.numpy.invkernel import InvKernelOp

def InvGaussKernel(D,Dv,sigma):
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
                 'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
                 'b = Vy(' + str(Dv) + ')',  # Third arg  : j-variable, of size Dv
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = InvKernelOp(formula, variables, 'b', axis=1)
    oos2 = np.array([1.0/sigma**2]).astype('float32')
    def Kinv(x,b):
        return my_routine(x,x,b,oos2)
    return Kinv
     
D = 2
N = 4
sigma = .1
x = np.random.rand(N, D).astype('float32')
b = np.random.rand(N, D).astype('float32')
Kinv = InvGaussKernel(D,D,sigma)
c = Kinv(x,b)
print("c = ",c)


