import numpy as np
from pykeops.numpy import Genred 

from pykeops.numpy.operations import InvKernelOp

D = 2
Dv = 2
N = 500
sigma = .1

# define the kernel : here a gaussian kernel
formula = 'Exp(-oos2*SqDist(x,y))*b'
variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
             'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
             'b = Vy(' + str(Dv) + ')',  # Third arg  : j-variable, of size Dv
             'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
             
# define the inverse kernel operation : here the 'b' argument specifies that linearity is with respect to variable b in formula.
Kinv = InvKernelOp(formula, variables, 'b')

# data
x = np.random.rand(N, D)
b = np.random.rand(N, D)
oos2 = np.array([1.0/sigma**2])

# apply
print("kernel inversion operation")
c = Kinv(x,x,b,oos2)
print("c = ",c)


