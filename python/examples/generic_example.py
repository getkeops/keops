import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..')

# Standard imports
import torch
from torch          import Tensor
from torch.autograd import Variable, grad
from bindings.torch.kernels import Kernel, kernel_product

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
use_cuda = torch.cuda.is_available()
use_cuda = False
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


# Convenience functions
def scal_to_var(x) :
    "Turns a float into a torch variable."
    return Variable(Tensor([x])).type(dtype)


def scalprod(x, y) :
    "Simple L2 scalar product."
    return torch.dot(x.view(-1), y.view(-1))

# Define our kernel
if True : # Standard kernel
    kernel = Kernel("gaussian(x,y)")
else : # Custom kernel, defined using the naming conventions of the libkp
    kernel              = Kernel()
    kernel.features     = "locations" # we could also use "locations+directions", etc.
    # Symbolic formula, for the libkp backend
    kernel.formula_sum  = "( -Cst(G)*SqDist(X,Y) )"
    # Pytorch routine, for the pytorch backend
    kernel.routine_sum  = lambda g=None, xmy2=None, **kwargs : \
                                  -g*xmy2

# Wrap it (and its parameters) into a JSON dict structure
sigma = scal_to_var(0.5)
params = {
    "id"      : kernel,
    "gamma"   : 1./sigma**2,
    "backend" : "auto",
}

# Define our dataset
npoints_x = 100
npoints_y = 700
dimpoints = 3
dimsignal = 2

a = Variable(torch.randn(npoints_x,dimsignal), requires_grad=True)
x = Variable(torch.randn(npoints_x,dimpoints), requires_grad=True)
y = Variable(torch.randn(npoints_y,dimpoints), requires_grad=True)
b = Variable(torch.randn(npoints_y,dimsignal), requires_grad=True)

# Test, using a pytorch or libkp backend
for backend in ["pytorch", "auto"] :
    params["backend"] = backend
    print("Backend :", backend, "--------------------------")
    Kxy_b  = kernel_product( x,y,b, params)
    aKxy_b = scalprod(a, Kxy_b)

    # Computing a gradient is that easy - we can also use the "aKxy_b.backward()" syntax. 
    # Notice the "create_graph=True", which will allow us to compute
    # higher order derivatives.
    [grad_x, grad_y]   = grad(aKxy_b, [x, y], create_graph=True)

    grad_x_norm        = scalprod(grad_x, grad_x)
    [grad_xx, grad_xy] = grad(grad_x_norm, [x,y])

    print("Kernel dot product  : ", aKxy_b )
    print("Gradient wrt. x     : ", grad_x[:3,:] )
    print("Gradient wrt. y     : ", grad_y[:3,:] )
    print("Arbitrary formula 1 : ", grad_xx[:3,:] )
    print("Arbitrary formula 2 : ", grad_xy[:3,:] )
