"""
LogSumExp reduction
==============================
"""

####################################################################
# Let's compute the (3000,1) tensor :math:`c` whose entries
# :math:`c_i` are given by:
#
# .. math::
#   c_i = \log \left[ \sum_j \exp\left( (p-a_j)^2 \exp(x_i+y_j) \right) \right]
#
# where
#
# * :math:`x` is a (3000,1) tensor, with entries :math:`x_i`.
# * :math:`y` is a (5000,1) tensor, with entries :math:`y_j`.
# * :math:`a` is a (5000,1) tensor, with entries :math:`a_j`.
# * :math:`p` is a scalar, encoded as a vector of size (1,).
#


####################################################################
# Setup
# -----
#
# Standard imports:

import time

import torch
from matplotlib import pyplot as plt
from torch.autograd import grad

from pykeops.torch import Genred

#####################################################################
# Declare random inputs:

M = 3000
N = 5000

dtype = "float32"  # Could be 'float32' or 'float64'
torchtype = torch.float32 if dtype == "float32" else torch.float64

x = torch.rand(M, 1, dtype=torchtype)
y = torch.rand(N, 1, dtype=torchtype, requires_grad=True)
a = torch.rand(N, 1, dtype=torchtype)
p = torch.rand(1, dtype=torchtype)

####################################################################
# Define a custom formula
# -----------------------

formula = "Square(p-a)*Exp(x+y)"
variables = [
    "x = Vi(1)",  # First arg   : i-variable, of size 1 (scalar)
    "y = Vj(1)",  # Second arg  : j-variable, of size 1 (scalar)
    "a = Vj(1)",  # Third arg   : j-variable, of size 1 (scalar)
    "p = Pm(1)",
]  # Fourth  arg : Parameter,  of size 1 (scalar)

start = time.time()

####################################################################
# Our log-sum-exp reduction is performed over the index :math:`j`,
# i.e. on the axis ``1`` of the kernel matrix.
# The output c is an :math:`x`-variable indexed by :math:`i`.

my_routine = Genred(formula, variables, reduction_op="LogSumExp", axis=1, dtype=dtype)
c = my_routine(x, y, a, p, backend="CPU")

# N.B.: By specifying backend='CPU', we can make sure that the result is computed using a simple C++ for loop.
print(
    "Time to compute the convolution operation on the cpu: ",
    round(time.time() - start, 5),
    "s",
    end=" ",
)

#######################################################################
# We compare with the unstable, naive computation "Log of Sum of Exp":

my_routine2 = Genred(
    "Exp(" + formula + ")", variables, reduction_op="Sum", axis=1, dtype=dtype
)
c2 = torch.log(my_routine2(x, y, a, p, backend="CPU"))
print("(relative error: ", ((c2 - c).norm() / c.norm()).item(), ")")

# Plot the results next to each other:
plt.plot(c.detach().cpu().numpy()[:40], "-", label="KeOps - Stable")
plt.plot(c2.detach().cpu().numpy()[:40], "--", label="KeOps - Unstable")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

####################################################################
# Compute the gradient
# --------------------
# Now, let's compute the gradient of :math:`c` with
# respect to :math:`y`. Since :math:`c` is not scalar valued,
# its "gradient" :math:`\partial c` should be understood as the adjoint of the
# differential operator, i.e. as the linear operator that:
#
# - takes as input a new tensor :math:`e` with the shape of :math:`c`
# - outputs a tensor :math:`g` with the shape of :math:`y`
#
# such that for all variation :math:`\delta y` of :math:`y` we have:
#
# .. math::
#
#    \langle \text{d} c . \delta y , e \rangle  =  \langle g , \delta y \rangle  =  \langle \delta y , \partial c . e \rangle
#
# Backpropagation is all about computing the tensor :math:`g=\partial c . e` efficiently, for arbitrary values of :math:`e`:

# Declare a new tensor of shape (M,1) used as the input of the gradient operator.
# It can be understood as a "gradient with respect to the output c"
# and is thus called "grad_output" in the documentation of PyTorch.
e = torch.rand_like(c)
# Call the gradient op:
start = time.time()
g = grad(c, y, e)[0]
# PyTorch remark : grad(c, y, e) alone outputs a length 1 tuple, hence the need for [0] at the end.

print(
    "Time to compute gradient of convolution operation on the cpu: ",
    round(time.time() - start, 5),
    "s",
    end=" ",
)

####################################################################
# We compare with gradient of Log of Sum of Exp:

g2 = grad(c2, y, e)[0]
print("(relative error: ", ((g2 - g).norm() / g.norm()).item(), ")")


# Plot the results next to each other:
plt.plot(g.detach().cpu().numpy()[:40], "-", label="KeOps - Stable")
plt.plot(g2.detach().cpu().numpy()[:40], "--", label="KeOps - Unstable")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

####################################################################
# Same operations performed on the Gpu
# ------------------------------------
#
# Of course, this will only work if you own a Gpu...

if torch.cuda.is_available():
    # first transfer data on gpu
    pc, ac, xc, yc, ec = p.cuda(), a.cuda(), x.cuda(), y.cuda(), e.cuda()
    # then call the operations
    start = time.time()
    c3 = my_routine(xc, yc, ac, pc, backend="GPU")
    print(
        "Time to compute convolution operation on the gpu:",
        round(time.time() - start, 5),
        "s ",
        end="",
    )
    print("(relative error:", float(torch.abs((c2 - c3.cpu()) / c2).mean()), ")")
    start = time.time()
    g3 = grad(c3, yc, ec)[0]
    print(
        "Time to compute gradient of convolution operation on the gpu:",
        round(time.time() - start, 5),
        "s ",
        end="",
    )
    print("(relative error:", float(torch.abs((g2 - g3.cpu()) / g2).mean()), ")")

    # Plot the results next to each other:
    plt.plot(c.detach().cpu().numpy()[:40], "-", label="KeOps - CPU")
    plt.plot(c3.detach().cpu().numpy()[:40], "--", label="KeOps - GPU")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # Plot the results next to each other:
    plt.plot(g.detach().cpu().numpy()[:40], "-", label="KeOps - CPU")
    plt.plot(g3.detach().cpu().numpy()[:40], "--", label="KeOps - GPU")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
