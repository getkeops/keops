"""
Example of KeOps reduction using the generic syntax. This example corresponds
to the one described in the documentation file generic_syntax.md. 

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

import torch
from torch.autograd import Variable, grad

import time

from pykeops.torch.generic_sum import GenericSum


#--------------------------------------------------------------#
#                   Define our dataset                         #
#--------------------------------------------------------------#

p = Variable(torch.randn(1,1), requires_grad=True)
a = Variable(torch.randn(5000,1), requires_grad=True)
x = Variable(torch.randn(3000,3), requires_grad=True)
y = Variable(torch.randn(5000,3), requires_grad=True)


#--------------------------------------------------------------#
#                        Kernel                                #
#--------------------------------------------------------------#

aliases = ["p=Pm(0,1)","a=Vy(1,1)","x=Vx(2,3)","y=Vy(3,3)"]
formula = "Square(p-a)*Exp(x+y)"
signature   =   [ (3, 0), (1, 2), (1, 1), (3, 0), (3, 1) ]
sum_index = 0       # 0 means summation over j, 1 means over i 
start = time.time()
c = GenericSum.apply("auto",aliases,formula,signature,sum_index,p,a,x,y)

print("time to compute convolution operation on cpu : ",round(time.time()-start,2)," seconds")


#--------------------------------------------------------------#
#                        Gradient                              #
#--------------------------------------------------------------#

# testing the gradient : we take the gradient with respect to y. In fact since 
# c is not scalar valued, "gradient" means in fact the adjoint of the differential
# operator, which is a linear operation that takes as input a new tensor with same
# size as c and outputs a tensor with same size as y

# new variable of size 3000x3 used as input of the gradient
e = Variable(torch.randn(3000,3), requires_grad=True)
# call to the gradient
start = time.time()
d = grad(c,y,e)[0]
# remark : grad(c,y,e) alone outputs a length 1 tuple, hence the need for [0] at the end.

print("time to compute gradient of convolution operation on cpu : ",round(time.time()-start,2)," seconds")



#--------------------------------------------------------------#
#            same operations performed on the Gpu              #
#--------------------------------------------------------------#

#  (this will of course only work if you have a Gpu)

if torch.cuda.is_available():
	# first transfer data on gpu
	p,a,x,y,e = p.cuda(), a.cuda(), x.cuda(), y.cuda(), e.cuda()
	# then call the operations
	start = time.time()
	c = GenericSum.apply("auto",aliases,formula,signature,sum_index,p,a,x,y)
	print("time to compute convolution operation on gpu : ",round(time.time()-start,2)," seconds")
	d = grad(c,y,e)[0]
	print("time to compute gradient of convolution operation on gpu : ",round(time.time()-start,2)," seconds")
