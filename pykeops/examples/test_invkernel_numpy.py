import numpy as np
import time 

from pykeops.numpy.operations import InvKernelOp

D = 2
Dv = 2
N = 100
sigma = .1

# define the kernel : here a gaussian kernel
formula = 'Exp(-oos2*SqDist(x,y))*b'
variables = ['x = Vx(' + str(D) + ')',  # First arg   : i-variable, of size D
             'y = Vy(' + str(D) + ')',  # Second arg  : j-variable, of size D
             'b = Vy(' + str(Dv) + ')',  # Third arg  : j-variable, of size Dv
             'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
             
# define the inverse kernel operation : here the 'b' argument specifies that linearity is with respect to variable b in formula.
lmbda = 0.01
Kinv = InvKernelOp(formula, variables, 'b', lmbda=lmbda, axis=1)

# data
x = np.random.rand(N, D)
b = np.random.rand(N, D)
oos2 = np.array([1.0/sigma**2])

# apply
print("Kernel inversion operation with gaussian kernel, ",N," points in dimension ",D)
start = time.time()
c = Kinv(x,x,b,oos2)
end = time.time()
print('Time to perform (KeOps):', round(end - start, 5), 's')

# compare with direct numpy implementation
start = time.time()
c_ = np.linalg.solve(lmbda*np.eye(N)+np.exp(-np.sum((x[:,None,:]-x[None,:,:])**2,axis=2)/sigma**2),b)
end = time.time()
print('Time to perform (Numpy):', round(end - start, 5), 's')
print("relative error = ",np.linalg.norm(c-c_)/np.linalg.norm(c_))


