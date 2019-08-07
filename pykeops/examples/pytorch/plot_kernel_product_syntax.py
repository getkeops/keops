"""
High-level kernel_product syntax
================================

Let's showcase KeOps's high-level interface on 3D point clouds.

 
"""

####################################################################
# Setup
# ------------------
#
# Standard imports:

import matplotlib.pyplot as plt
import torch
from torch.autograd import grad

from pykeops.torch import Kernel, kernel_product


####################################################################
# Convenience functions:

def scal_to_var(x):
    """Turns a float into a length-1 tensor, that can be used as a parameter."""
    return torch.tensor([x], requires_grad=True, device=device)


def scalprod(x, y):
    """Simple L2 scalar product."""
    return torch.dot(x.view(-1), y.view(-1))


def disp(x):
    """Returns a printable version of x."""
    return x.detach().cpu().numpy()


#####################################################################
# Declare random inputs:

npoints_x, npoints_y = 100, 700
dimpoints, dimsignal = 3, 1

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# N.B.: PyTorch's default dtype is float32
a = torch.randn(npoints_x, dimsignal, device=device)
x = torch.randn(npoints_x, dimpoints, requires_grad=True, device=device)
y = torch.randn(npoints_y, dimpoints, device=device)
b = torch.randn(npoints_y, dimsignal, device=device)

####################################################################
# A Gaussian convolution
# ----------------------

# Wrap the kernel's parameters into a JSON dict structure
sigma = scal_to_var(-1.5)
params = {
    'id': Kernel('gaussian(x,y)'),
    'gamma': .5 / sigma ** 2,
}

####################################################################
# Test, using both **PyTorch** and **KeOps** (online) backends:
axes = plt.subplot(2,1,1), plt.subplot(2,1,2)

for backend, linestyle, label in [("auto",    "-", "KeOps"),
                                  ("pytorch", "--", "PyTorch")]:
    
    Kxy_b = kernel_product(params, x, y, b, backend=backend)
    aKxy_b = scalprod(a, Kxy_b)
    
    # Computing a gradient is that easy - we can also use the 'aKxy_b.backward()' syntax.
    # Notice the 'create_graph=True', which will allow us to compute
    # higher order derivatives.
    [grad_x, grad_s] = grad(aKxy_b, [x, sigma], create_graph=True)

    grad_x_norm = scalprod(grad_x, grad_x)
    [grad_xx] = grad(grad_x_norm, [x], create_graph=True)

    print("Backend = {:^7}:  cost = {:.4f}, grad wrt. s = {:.4f}".format(
            label, aKxy_b.item(), grad_s.item() ))


    # Fancy display: plot the results next to each other.
    axes[0].plot(grad_x.detach().cpu().numpy()[:40,0], linestyle, label=label)
    axes[0].legend(loc='lower right')

    axes[1].plot(grad_xx.detach().cpu().numpy()[:40,2], linestyle, label=label)
    axes[1].legend(loc='lower right')

plt.tight_layout() ; plt.show()



####################################################################
# Custom kernel formula
# -------------------------
#
# Through a direct access to :class:`pykeops.torch.Formula`
# and the dict :mod:`pykeops.torch.kernel_formulas`,
# users may add their own formulas to the :mod:`pykeops.torch.Kernel` parser.
#
# .. warning::
#   As of today, this syntax is only **loosely supported**:
#   the design of KeOps's high-level interface will depend
#   on user feedbacks, and may evolve in the coming months.

from pykeops.torch import Formula, kernel_formulas

print('Formulas supported out-of-the-box: ', kernel_formulas.keys())

kernel_formulas['my_formula'] = Formula( 
    # Symbolic formulas, for the KeOps backend:
    formula_sum='Exp( ({X}|{Y}) - WeightedSqDist({G},{X},{Y}) )',
    formula_log='( ({X}|{Y}) - WeightedSqDist({G},{X},{Y}) )',

    # Pytorch routines, for the 'pure pytorch' backend:
    routine_sum=lambda gxmy2=None, xsy=None, **kwargs: (xsy - gxmy2).exp(),
    routine_log=lambda gxmy2=None, xsy=None, **kwargs:  xsy - gxmy2,
)

print('After a dynamic addition: ', kernel_formulas.keys())
kernel = Kernel('my_formula(x,y)')

####################################################################
# Wrap it (and its parameters) into a JSON dict structure
sigma = scal_to_var(0.5)
params = {
    'id': kernel,
    'gamma': .5 / sigma ** 2,
}

####################################################################
# Test our new kernel, using **PyTorch** and **KeOps** (online) backends:


for backend, linestyle, label in [("auto",    "-", "KeOps"),
                                  ("pytorch", "--", "PyTorch")]:

    # For a change, let's use the LogSumExp mode:                              
    Kxy_b = kernel_product(params, x, y, b, backend=backend, mode="lse")
    aKxy_b = scalprod(a, Kxy_b)
    
    # Computing a gradient is that easy - we can also use the 'aKxy_b.backward()' syntax.
    # Notice the 'create_graph=True', which will allow us to compute
    # higher order derivatives.
    [grad_x, grad_s] = grad(aKxy_b, [x, sigma], create_graph=True)


    print("Backend = {:^7}:  cost = {:.4f}, grad wrt. s = {:.4f}".format(
            label, aKxy_b.item(), grad_s.item() ))

    # Fancy display: plot the results next to each other.
    plt.plot(grad_x.detach().cpu().numpy()[:40,0], linestyle, label=label)
    plt.legend(loc='lower right')

plt.tight_layout() ; plt.show()
