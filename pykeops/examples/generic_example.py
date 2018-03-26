import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

# Standard imports
import torch
from torch          import Tensor
from torch.autograd import Variable, grad
from pykeops.torch.kernels import Kernel, kernel_product

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


# Convenience functions
def scal_to_var(x) :
    "Turns a float into a torch variable."
    return Variable(Tensor([x]), requires_grad=True).type(dtype)

def scalprod(x, y) :
    "Simple L2 scalar product."
    return torch.dot(x.view(-1), y.view(-1))


# Wrap the kernel's parameters into a JSON dict structure
sigma = scal_to_var(0.5)
params = {
    "id"      : Kernel("gaussian(x,y)"),
    "gamma"   : 1./sigma**2
}

# Define our dataset
npoints_x = 100 ; npoints_y = 700
dimpoints = 3   ; dimsignal = 1

a = Variable(torch.randn(npoints_x,dimsignal), requires_grad=True).type(dtype)
x = Variable(torch.randn(npoints_x,dimpoints), requires_grad=True).type(dtype)
y = Variable(torch.randn(npoints_y,dimpoints), requires_grad=True).type(dtype)
b = Variable(torch.randn(npoints_y,dimsignal), requires_grad=True).type(dtype)

# N.B.: "sum" is default, "log" is for "log-sum-exp"
modes = ["sum", "log"] if dimsignal==1 else ["sum"]

# Test, using a pytorch or libkp backend
for mode in modes : 
    print("Mode :", mode, "========================================")
    for backend in ["auto"] :
        params["backend"] = backend
        print("Backend :", backend, "--------------------------")
        Kxy_b  = kernel_product( x,y,b, params, mode=mode)
        aKxy_b = scalprod(a, Kxy_b)

        print("Kernel dot product  : ", aKxy_b )

        # Computing a gradient is that easy - we can also use the "aKxy_b.backward()" syntax. 
        # Notice the "create_graph=True", which will allow us to compute
        # higher order derivatives.
        [grad_x, grad_y, grad_s]   = grad(aKxy_b, [x, y, sigma], create_graph=True)

        print("Gradient wrt. x     : ", grad_x[:2,:] )
        print("Gradient wrt. y     : ", grad_y[:2,:] )
        print("Gradient wrt. s     : ", grad_s       )

        grad_x_norm        = scalprod(grad_x, grad_x)
        [grad_xx, grad_xy] = grad(grad_x_norm, [x,y], create_graph=True)

        print("Arbitrary formula 1 : ", grad_xx[:2,:] )
        print("Arbitrary formula 2 : ", grad_xy[:2,:] )
        
        grad_s_norm        = scalprod(grad_s, grad_s)
        [grad_sx, grad_ss] = grad(grad_s_norm, [x,sigma], create_graph=True)

        print("Arbitrary formula 3 : ", grad_sx[:2,:] )
        print("Arbitrary formula 4 : ", grad_ss       )
