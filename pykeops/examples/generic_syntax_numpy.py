"""
Example of keops reduction using the generic syntax. This example corresponds
to the one described in the documentation file generic_syntax.md . It uses
pure numpy framework (no pytorch).

# this example computes the following tensor operation :
# inputs :
#   p   : a scalar (antered as a 1x1 tensor)
#   a   : a 5000x1 tensor, with entries denoted a_j 
#   x   : a 3000x3 tensor, with entries denoted x_i^u
#   y   : a 5000x3 tensor, with entries denoted y_j^u
# output :
#   c   : a 3000x3 tensor, with entries denoted c_i^u, such that
#   c_i^u = sum_j (p-y_j)^2 exp(a_i^u+b_j^u)

"""

#--------------------------------------------------------------#
#                     Standard imports                         #
#--------------------------------------------------------------#
import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import time
import numpy as np

from pykeops.numpy.generic_sum import GenericSum_np


#--------------------------------------------------------------#
#                   Define our dataset                         #
#--------------------------------------------------------------#

p = np.random.randn(1,1).astype('float32')
a = np.random.randn(5000,1).astype('float32')
x = np.random.randn(3000,3).astype('float32')
y = np.random.randn(5000,3).astype('float32')


#--------------------------------------------------------------#
#                        Kernel                                #
#--------------------------------------------------------------#

aliases = ["p=Pm(0,1)","a=Vy(1,1)","x=Vx(2,3)","y=Vy(3,3)"]
formula = "Square(p-a)*Exp(x+y)"
signature   =   [ (3, 0), (1, 2), (1, 1), (3, 0), (3, 1) ]
sum_index = 0       # 0 means summation over j, 1 means over i 

start = time.time()
c = GenericSum_np("auto",aliases,formula,signature,sum_index,p,a,x,y)

print("time to compute c on cpu : ",round(time.time()-start,2)," seconds")

#--------------------------------------------------------------#
#                        Gradient                              #
#--------------------------------------------------------------#

# testing the gradient : we take the gradient with respect to y. In fact since 
# c is not scalar valued, "gradient" means in fact the adjoint of the differential
# operator, which is a linear operation that takes as input a new tensor with same
# size as c and outputs a tensor with same size as y

# new variable of size 3000x3 used as input of the gradient
e = np.random.randn(3000,3).astype('float32')


aliases_grad = ["p=Pm(0,1)","a=Vy(1,1)","x=Vx(2,3)","y=Vy(3,3)","e=Vx(4,3)"]
formula_grad = "Grad(Square(p-a)*Exp(x+y),y,e)"
signature_grad   =   [ (3, 1), (1, 2), (1, 1), (3, 0), (3, 1) , (3, 0) ]
sum_index = 1       # 0 means summation over j, 1 means over i 


# call to the gradient
start = time.time()
d = GenericSum_np("auto",aliases_grad,formula_grad,signature_grad,sum_index,p,a,x,y,e)
print("time to compute d on cpu : ",round(time.time()-start,2)," seconds")
