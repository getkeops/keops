"""
==========================================
Fancy reductions, solving linear systems
==========================================


"""

#############################################################
# As discussed in the previous notebook,
# KeOps :class:`LazyTensors <pykeops.torch.LazyTensor>`
# support a wide range of **mathematical formulas**.
# Let us now discuss the different operators that can be used
# to **reduce** our large M-by-N symbolic tensors into
# vanilla NumPy arrays or PyTorch tensors.
#
# .. note::
#   In this tutorial, we stick to the **PyTorch** interface;
#   but note that apart from a few lines on backpropagation,
#   everything here can be seamlessly translated to vanilla **NumPy+KeOps** code.
#
# LogSumExp, KMin and advanced reductions
# ---------------------------------------------------
#
# First, let's build some large :class:`LazyTensors <pykeops.torch.LazyTensor>`
# ``S_ij`` and ``V_ij`` which respectively handle **scalar** and **vector**
# formulas:

import torch

from pykeops.torch import LazyTensor

use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
M, N = (100000, 200000) if use_cuda else (1000, 2000)
D = 3

x = torch.randn(M, D).type(tensor)
y = torch.randn(N, D).type(tensor)

x_i = LazyTensor(x[:, None, :])  # (M, 1, D) LazyTensor
y_j = LazyTensor(y[None, :, :])  # (1, N, D) LazyTensor

V_ij = x_i - y_j  # (M, N, D) symbolic tensor of differences
S_ij = (V_ij ** 2).sum(-1)  # (M, N, 1) = (M, N) symbolic matrix of squared distances

print(S_ij)
print(V_ij)

############################################################################
# As we've seen earlier, the :meth:`pykeops.torch.LazyTensor.sum()` reduction can be used
# on both ``S_ij`` and ``V_ij`` to produce genuine PyTorch 2D tensors:

print("Sum reduction of S_ij wrt. the 'N' dimension:", S_ij.sum(dim=1).shape)

###############################################
# Note that :class:`LazyTensors<pykeops.torch.LazyTensor>`
# support reductions over both indexing dimensions ``M`` and ``N``,
# which can be specified using the PyTorch ``dim``
# or the NumPy ``axis`` optional arguments:

print("Sum reduction of V_ij wrt. the 'M' dimension:", V_ij.sum(axis=0).shape)

##################################################################
# Just like PyTorch tensors,
# :mod:`pykeops.torch.LazyTensor`
# support a **stabilized** `log-sum-exp reduction <https://en.wikipedia.org/wiki/LogSumExp>`_,
# computed efficiently with a **running maximum** in the CUDA loop. For example, the
# following line computes :math:`\log(\sum_ie^{S_{ij}})`

print(
    "LogSumExp reduction of S_ij wrt. the 'M' dimension:", S_ij.logsumexp(dim=0).shape
)

##################################################################
# This reduction supports a weight parameter that can be scalar or vector-valued.
# For example, the following line computes :math:`\log(\sum_je^{S_{ij}}V_{ij})`
print(
    "LogSumExp reduction of S_ij, with 'weight' V_ij, wrt. the 'N' dimension:",
    S_ij.logsumexp(dim=1, weight=V_ij).shape,
)

###############################################################
# Going further, the :meth:`pykeops.torch.LazyTensor.min`, :meth:`pykeops.torch.LazyTensor.max()`, :meth:`pykeops.torch.LazyTensor.argmin` or :meth:`pykeops.torch.LazyTensor.argmax`
# reductions work as expected, following the sensible NumPy convention:

print("Min    reduction of S_ij wrt. the 'M' dimension:", S_ij.min(dim=0).shape)
print("ArgMin reduction of S_ij wrt. the 'N' dimension:", S_ij.argmin(dim=1).shape)

print("Max    reduction of V_ij wrt. the 'M' dimension:", V_ij.max(dim=0).shape)
print("ArgMax reduction of V_ij wrt. the 'N' dimension:", V_ij.argmax(dim=1).shape)

##################################################################
# To compute both quantities in a single pass, feel free to use
# the :meth:`pykeops.torch.LazyTensor.min_argmin` and :meth:`pykeops.torch.LazyTensor.max_argmax` reductions:

m_i, s_i = S_ij.min_argmin(dim=0)
print("Min-ArgMin reduction on S_ij wrt. the 'M' dimension:", m_i.shape, s_i.shape)

m_i, s_i = V_ij.max_argmax(dim=1)
print("Max-ArgMax reduction on V_ij wrt. the 'N' dimension:", m_i.shape, s_i.shape)

##################################################################
# More interestingly, KeOps also provides support for
# the :meth:`pykeops.torch.LazyTensor.Kmin`, :meth:`pykeops.torch.LazyTensor.argKmin` and :meth:`pykeops.torch.LazyTensor.Kmin_argKmin`
# reductions that can be used to implement an efficient
# :doc:`K-nearest neighbor algorithm <../knn/plot_knn_torch>` :
#

K = 5
print("KMin    reduction of S_ij wrt. the 'M' dimension:", S_ij.Kmin(K=K, dim=0).shape)
print(
    "ArgKMin reduction of S_ij wrt. the 'N' dimension:", S_ij.argKmin(K=K, dim=1).shape
)

##################################################################
# It even works on vector formulas!

K = 7
print("KMin    reduction of V_ij wrt. the 'M' dimension:", V_ij.Kmin(K=K, dim=0).shape)
print(
    "ArgKMin reduction of V_ij wrt. the 'N' dimension:", V_ij.argKmin(K=K, dim=1).shape
)

#################################################################
# Finally, the :meth:`pykeops.torch.LazyTensor.sumsoftmaxweight` reduction
# can be used to computed weighted SoftMax combinations
#
# .. math::
#   a_i = \frac{\sum_j \exp(s_{i,j})\,v_{i,j} }{\sum_j \exp(s_{i,j})},
#
# with scalar coefficients :math:`s_{i,j}` and arbitrary vector weights :math:`v_{i,j}`:

a_i = S_ij.sumsoftmaxweight(V_ij, dim=1)
print(
    "SumSoftMaxWeight reduction of S_ij, with weights V_ij, wrt. the 'N' dimension:",
    a_i.shape,
)

#################################################################
# Solving linear systems
# -------------------------------------
#
# Inverting large M-by-M linear systems is a fundamental problem in applied mathematics.
# To help you solve problems of the form
#
# .. math::
#       & & a^{\star} & =\operatorname*{argmin}_a  \| (\alpha\operatorname{Id}+K_{xx})a -b\|^2_2, \\\\
#       &\text{i.e.}\quad &  a^{\star} & = (\alpha \operatorname{Id} + K_{xx})^{-1}  b,
#
# KeOps :mod:`pykeops.torch.LazyTensor` support
# a simple :meth:`LazyTensor.solve(b, alpha=1e-10)<pykeops.torch.LazyTensor.solve>` operation that we use as follows:

x = torch.randn(M, D, requires_grad=True).type(tensor)  # Random point cloud
x_i = LazyTensor(x[:, None, :])  # (M, 1, D) LazyTensor
x_j = LazyTensor(x[None, :, :])  # (1, M, D) LazyTensor

K_xx = (-((x_i - x_j) ** 2).sum(-1)).exp()  # Symbolic (M, M) Gaussian kernel matrix

alpha = 0.1  # "Ridge" regularization parameter
b_i = torch.randn(M, 4).type(tensor)  # Target signal, supported by the x_i's
a_i = K_xx.solve(b_i, alpha=alpha)  # Source signal, supported by the x_i's

print("a_i is now a {} of shape {}.".format(type(a_i), a_i.shape))

##################################################################
# As expected, we can now check that:
#
# .. math::
#   (\alpha \operatorname{Id} + K_{xx}) \,a \simeq b.
#

c_i = alpha * a_i + K_xx @ a_i  # Reconstructed target signal

print("Mean squared reconstruction error: {:.2e}".format(((c_i - b_i) ** 2).mean()))

#################################################################
# Please note that just like (nearly) all the other :class:`LazyTensor <pykeops.torch.LazyTensor>` methods,
# :meth:`pykeops.torch.LazyTensor.solve` fully supports the :mod:`torch.autograd` module:

[g_i] = torch.autograd.grad((a_i ** 2).sum(), [x])

print("g_i is now a {} of shape {}.".format(type(g_i), g_i.shape))

#################################################################
# .. warning::
#   As of today, the :meth:`pykeops.torch.LazyTensor.solve` operator only implements
#   a `conjugate gradient descent <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_
#   under the assumption that **K_xx is a symmetric, positive-definite matrix**.
#   To solve generic systems, you could either
#   :doc:`interface KeOps with the routines of the SciPy package <../backends/plot_scipy>`
#   or implement your own solver, mimicking our
#   `reference implementation. <https://github.com/getkeops/keops/blob/master/pykeops/common/operations.py>`_
