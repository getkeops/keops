r"""
================================================
A wrapper for NumPy and PyTorch arrays
================================================

KeOps brings **semi-symbolic** calculus
to modern computing libraries:
it alleviates the need for **huge intermediate variables**
such as *kernel* or *distance* matrices in machine
learning and computational geometry.
 
"""

#########################################################################
# First steps
# -----------
#
# A simple interface to the KeOps inner routines is provided by
# the :class:`pykeops.numpy.LazyTensor` or :class:`pykeops.torch.LazyTensor`
# **symbolic wrapper**, to be used with **NumPy arrays** or **PyTorch
# tensors** respectively.
#
# To illustrate its main features on a **simple example**, let's generate two point
# clouds :math:`(x_i)_{i\in[1,M]}`  and :math:`(y_j)_{j\in[1,N]}` in the unit square:

import numpy as np

M, N = 1000, 2000
x = np.random.rand(M, 2)
y = np.random.rand(N, 2)

##########################################################################
# With NumPy, an efficient way of computing the index of the **nearest y-neighbor**
#
# .. math::
#   \sigma(i) = \arg \min_{j\in [1,N]} \| x_i - y_j\|^2
#
# for all points :math:`x_i` is to perform a :func:`numpy.argmin()`
# reduction on the **M-by-N matrix** of squared distances
#
# .. math::
#   D_{i,j} = \|x_i-y_j\|^2,
#
# computed using **tensorized**, broadcasted operators:
#

x_i = x[:, None, :]  # (M, 1, 2) numpy array
y_j = y[None, :, :]  # (1, N, 2) numpy array

D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) array of squared distances |x_i-y_j|^2
s_i = np.argmin(D_ij, axis=1)  # (M,)   array of integer indices
print(s_i[:10])

###########################################################################
# That's good! Going further, we can speed-up these computations
# using the **CUDA routines** of the PyTorch library:

import torch

use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

x_i = tensor(x[:, None, :])  # (M, 1, 2) torch tensor
y_j = tensor(y[None, :, :])  # (1, N, 2) torch tensor

D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) tensor of squared distances |x_i-y_j|^2
s_i = D_ij.argmin(dim=1)  # (M,)   tensor of integer indices
print(s_i[:10])

###########################################################################
# But **can we scale to larger point clouds?**
# Unfortunately, tensorized codes will throw an exception
# as soon as the **M-by-N matrix** :math:`(D_{i,j})` stops fitting
# contiguously on the device memory. This generally happens
# when :math:`\sqrt{MN}` goes past a hardware-dependent threshold
# in the [5,000; 50,000] range:

M, N = (100000, 200000) if use_cuda else (1000, 2000)
x = np.random.rand(M, 2)
y = np.random.rand(N, 2)

x_i = tensor(x[:, None, :])  # (M, 1, 2) torch tensor
y_j = tensor(y[None, :, :])  # (1, N, 2) torch tensor

try:
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) tensor of squared distances |x_i-y_j|^2
except RuntimeError as err:
    print(err)

###########################################################################
# **That's unfortunate...** And unexpected!
# After all, modern GPUs routinely handle
# `the real-time rendering of scenes with millions of triangles moving around <https://www.youtube.com/watch?v=RaLQuJtQ-gc>`_.
# So how do `graphics <https://www.siggraph.org/>`_ programmers achieve such
# a level of performance?
#
# The key to efficient numerical schemes is to remark that
# even though the distance matrix :math:`(D_{i,j})` is not **sparse** in the
# traditional sense, it definitely is **compact from a computational perspective**.
# Since its coefficients are fully described by two lists of points
# and a **symbolic formula**, sensible implementations should
# compute required values on-the-fly...
# and bypass, **lazily**, the cumbersome pre-computation and storage
# of all pairwise distances :math:`\|x_i-y_j\|^2`.
#
#

from pykeops.numpy import LazyTensor as LazyTensor_np

x_i = LazyTensor_np(
    x[:, None, :]
)  # (M, 1, 2) KeOps LazyTensor, wrapped around the numpy array x
y_j = LazyTensor_np(
    y[None, :, :]
)  # (1, N, 2) KeOps LazyTensor, wrapped around the numpy array y

D_ij = ((x_i - y_j) ** 2).sum(-1)  # **Symbolic** (M, N) matrix of squared distances
print(D_ij)

##########################################################
# With KeOps, implementing **lazy** numerical schemes really
# is **that simple**!
# Our :class:`LazyTensor<pykeops.torch.LazyTensor>` variables
# are encoded as a list of data arrays plus an arbitrary
# symbolic formula, written with a :doc:`custom mathematical syntax <../../api/math-operations>`
# that is modified after each "pythonic" operation such as ``-``, ``**2`` or ``.exp()``.
#
# We can then perform a :meth:`pykeops.torch.LazyTensor.argmin` reduction with
# an efficient Map-Reduce scheme, implemented
# as a `templated CUDA kernel <https://github.com/getkeops/keops/blob/master/keops/core/GpuConv1D.cu>`_ around
# our custom formula.
# As evidenced by our :doc:`benchmarks <../../_auto_benchmarks/index>`,
# the KeOps routines have a **linear memory footprint**
# and generally **outperform tensorized GPU implementations by two orders of magnitude**.

s_i = D_ij.argmin(dim=1).ravel()  # genuine (M,) array of integer indices
print("s_i is now a {} of shape {}.".format(type(s_i), s_i.shape))
print(s_i[:10])

##########################################################
# Going further, we can combine :class:`LazyTensors <pykeops.torch.LazyTensor>`
# using a **wide range of mathematical operations**.
# For instance, with data arrays stored directly on the GPU,
# an exponential kernel dot product
#
# .. math::
#   a_i = \sum_{j=1}^N \exp(-\|x_i-y_j\|)\cdot b_j
#
# in dimension D=10 can be performed with:

from pykeops.torch import LazyTensor

D = 10
x = torch.randn(M, D).type(tensor)  # M target points in dimension D, stored on the GPU
y = torch.randn(N, D).type(tensor)  # N source points in dimension D, stored on the GPU
b = torch.randn(N, 4).type(
    tensor
)  # N values of the 4D source signal, stored on the GPU

x.requires_grad = True  # In the next section, we'll compute gradients wrt. x!

x_i = LazyTensor(x[:, None, :])  # (M, 1, D) LazyTensor
y_j = LazyTensor(y[None, :, :])  # (1, N, D) LazyTensor

D_ij = ((x_i - y_j) ** 2).sum(-1).sqrt()  # Symbolic (M, N) matrix of distances
K_ij = (-D_ij).exp()  # Symbolic (M, N) Laplacian (aka. exponential) kernel matrix
a_i = K_ij @ b  # The matrix-vector product "@" can be used on "raw" PyTorch tensors!

print("a_i is now a {} of shape {}.".format(type(a_i), a_i.shape))

#############################################################################
#
# .. note::
#   KeOps LazyTensors have two symbolic or
#   "virtual" axes at positions -3 and -2.
#   Operations on the last "vector" dimension (-1)
#   or on optional "batch" dimensions (-4 and beyond)
#   are evaluated **lazily**.
#   On the other hand, a reduction on one of the two symbolic axes
#   (-2 or -3) triggers an **explicit computation**:
#   we return a standard dense array with no symbolic axes.
#
#
#
# Automatic differentiation
# -----------------------------------------------
#
# KeOps
# **fully support** the :mod:`torch.autograd` engine:
# we can backprop through KeOps reductions as easily as through
# vanilla PyTorch operations.
# For instance, coming back to the kernel dot product above,
# we can compute the gradient
#
# .. math::
#   g_i ~=~ \frac{\partial \sum_i \|a_i\|^2}{\partial x_i}
#
# with:

[g_i] = torch.autograd.grad((a_i ** 2).sum(), [x], create_graph=True)
print("g_i is now a {} of shape {}.".format(type(g_i), g_i.shape))

#############################################################################
# As usual with PyTorch, having set the ``create_graph=True`` option
# allows us to compute higher-order derivatives as needed:

[h_i] = torch.autograd.grad(g_i.exp().sum(), [x], create_graph=True)
print("h_i is now a {} of shape {}.".format(type(h_i), h_i.shape))

############################################################################
# .. warning::
#   As of today, backpropagation is **not supported** through
#   the :meth:`pykeops.torch.LazyTensor.min`, :meth:`pykeops.torch.LazyTensor.max` or :meth:`pykeops.torch.LazyTensor.Kmin` reductions:
#   we're working on it, but are not there just yet.
#   Until then, a simple workaround is to use
#   the indices computed by the
#   :meth:`pykeops.torch.LazyTensor.argmin`, :meth:`pykeops.torch.LazyTensor.argmax` or :meth:`pykeops.torch.LazyTensor.argKmin`
#   reductions to define a fully differentiable PyTorch tensor as we now explain.
#
# Coming back to our example about nearest neighbors in the unit cube:

x = torch.randn(M, 3).type(tensor)
y = torch.randn(N, 3).type(tensor)
x.requires_grad = True

x_i = LazyTensor(x[:, None, :])  # (M, 1, 3) LazyTensor
y_j = LazyTensor(y[None, :, :])  # (1, N, 3) LazyTensor
D_ij = ((x_i - y_j) ** 2).sum(-1)  # Symbolic (M, N) matrix of squared distances

#####################################################################
# We could compute the ``(M,)`` vector of squared distances to the **nearest y-neighbor** with:

to_nn = D_ij.min(dim=1).view(-1)

################################################################
# But instead, using:

s_i = D_ij.argmin(dim=1).view(-1)  # (M,) integer Torch tensor
to_nn_alt = ((x - y[s_i, :]) ** 2).sum(-1)

##########################################################
# outputs the same result, while also allowing us to **compute arbitrary gradients**:

print(
    "Difference between the two vectors: {:.2e}".format((to_nn - to_nn_alt).abs().max())
)

[g_i] = torch.autograd.grad(to_nn_alt.sum(), [x])
print("g_i is now a {} of shape {}.".format(type(g_i), g_i.shape))

###########################################################
# The only real downside here is that we had to write **twice** the
# "squared distance" formula that specifies our computation.
# We hope to fix this (minor) inconvenience sooner rather than later!
#


#############################################################################
# Batch processing
# -----------------------------------------------
#
# As should be expected, :class:`LazyTensors<pykeops.torch.LazyTensor>`
# also provide full support of **batch processing**,
# with broadcasting over dummy (=1) batch dimensions:

A, B = 7, 3  # Batch dimensions

x_i = LazyTensor(torch.randn(A, B, M, 1, D))
l_i = LazyTensor(torch.randn(1, 1, M, 1, D))
y_j = LazyTensor(torch.randn(1, B, 1, N, D))
s = LazyTensor(torch.rand(A, 1, 1, 1, 1))

D_ij = ((l_i * x_i - y_j) ** 2).sum(-1)  # Symbolic (A, B, M, N, 1) LazyTensor
K_ij = -1.6 * D_ij / (1 + s ** 2)  # Some arbitrary (A, B, M, N, 1) Kernel matrix

a_i = K_ij.sum(dim=3)
print("a_i is now a {} of shape {}.".format(type(a_i), a_i.shape))

##################################################################
# Everything works just fine, with two major caveats:
#
# - The structure of KeOps computations is still a little bit **rigid**:
#   :class:`LazyTensors<pykeops.torch.LazyTensor>` should only
#   be used in situations where the **large** dimensions M and N
#   over which the main reduction
#   is performed are in positions
#   -3 and -2 (respectively), with **vector** variables in position
#   -1 and an arbitrary number of batch dimensions beforehand.
#   We're working towards a full support of **tensor** variables,
#   but this will probably take some time to implement and test properly...
#
# - KeOps :class:`LazyTensors<pykeops.torch.LazyTensor>` never collapse
#   their last "dimension", even after a ``.sum(-1)`` reduction
#   whose **keepdim** argument is implicitely set to **True**.

print("Convenient, numpy-friendly shape:       ", K_ij.shape)
print("Actual shape, used internally by KeOps: ", K_ij._shape)

##################################################################
# This is the reason why in the example above,
# **a_i** is a 4D Tensor of shape ``(7, 3, 1000, 1)`` and **not**
# a 3D Tensor of shape ``(7, 3, 1000)``.
#


#############################################################################
# Supported formulas
# ------------------------------------
#
# The full range of mathematical operations supported by
# :class:`LazyTensors<pykeops.torch.LazyTensor>` is described
# in our API documentation.
# Let's just mention that the lines below define valid computations:
#

x_i = LazyTensor(torch.randn(A, B, M, 1, D))
l_i = LazyTensor(torch.randn(1, 1, M, 1, D))
y_j = LazyTensor(torch.randn(1, B, 1, N, D))
s = LazyTensor(torch.rand(A, 1, 1, 1, 1))

F_ij = (
    (x_i ** 1.5 + y_j / l_i).cos() - (x_i | y_j) + (x_i[:, :, :, :, 2] * s.relu() * y_j)
)
print(F_ij)

a_j = F_ij.sum(dim=2)
print("a_j is now a {} of shape {}.".format(type(a_j), a_j.shape))

#############################################################################
# Enjoy! And feel free to check the next tutorial for a discussion
# of the varied reduction operations that can be applied to
# KeOps :class:`LazyTensors<pykeops.torch.LazyTensor>`.
