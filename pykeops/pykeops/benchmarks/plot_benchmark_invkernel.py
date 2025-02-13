"""
Solving positive definite linear systems
=========================================

This benchmark compares the performances of KeOps versus Numpy and Pytorch on a inverse matrix operation. It uses the functions :class:`torch.KernelSolve <pykeops.torch.KernelSolve>` (see also :doc:`here <../_auto_examples/pytorch/plot_test_invkernel_torch>`) and  :class:`numpy.KernelSolve <pykeops.numpy.KernelSolve>` (see also :doc:`here <../_auto_examples/numpy/plot_test_invkernel_numpy>`).

In a nutshell, given :math:`x \in\mathbb R^{N\\times D}`  and :math:`b \in \mathbb R^{N\\times D_v}`, we compute :math:`a \in \mathbb R^{N\\times D_v}` so that

.. math::

  b = (\\alpha\operatorname{Id} + K_{x,x}) a \quad \Leftrightarrow \quad a = (\\alpha\operatorname{Id}+ K_{x,x})^{-1} b

where :math:`K_{x,x} = \Big[\exp(-\|x_i -x_j\|^2 / \sigma^2)\Big]_{i,j=1}^N`. The method is based on a conjugate gradient scheme. The benchmark tests various values of :math:`N \in [10, \cdots,10^6]`.


.. note::
    In this demo, we implement the linear operator :math:`K_xx`
    using a **bruteforce** implementation and do not leverage any multiscale
    or low-rank (Nystroem/multipole) decomposition of the Kernel matrix.
    Going further, advanced strategies and solvers
    are now available through the
    `GPyTorch <https://docs.gpytorch.ai/en/v1.1.1/examples/02_Scalable_Exact_GPs/KeOps_GP_Regression.html>`_
    and `Falkon <https://falkonml.github.io/falkon/>`_ libraries,
    which rely on a KeOps backend whenever relevant.


"""

#####################################################################
# Setup
# -----
# Standard imports:

import numpy as np
import torch
from matplotlib import pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import aslinearoperator, cg

from pykeops.numpy import KernelSolve as KernelSolve_np, LazyTensor
from pykeops.torch import KernelSolve
from pykeops.torch.utils import squared_distances as sqdist_torch
from pykeops.numpy.utils import squared_distances as sqdist_np

from benchmark_utils import flatten, random_normal, unit_tensor, full_benchmark

use_cuda = torch.cuda.is_available()

if torch.__version__ >= "1.8":
    torchsolve = lambda A, B: torch.linalg.solve(A, B)
else:
    torchsolve = lambda A, B: torch.solve(B, A)[0]

#####################################################################
# Benchmark specifications:
#

D = 3  # Let's do this in 3D
Dv = 1  # Dimension of the vectors (= number of linear problems to solve)

# Numbers of samples that we'll loop upon:
problem_sizes = flatten(
    [[1 * 10**k, 2 * 10**k, 5 * 10**k] for k in [1, 2, 3, 4, 5]] + [[10**6]]
)
D = 3  # We work with 3D points
Dv = 1  # and solve one problem at a time.


#####################################################################
# Create some random input data:
#


def generate_samples(N, device="cuda", lang="torch", **kwargs):
    """Generates a point cloud x, a scalar signal b of size N and two regularization parameters.

    Args:
        N (int): number of point.
        device (str, optional): "cuda", "cpu", etc. Defaults to "cuda".
        lang (str, optional): "torch", "numpy", etc. Defaults to "torch".

    Returns:
        3-uple of arrays: x, y, b
    """
    randn = random_normal(device=device, lang=lang)
    ones = unit_tensor(device=device, lang=lang)

    x = randn((N, D))
    b = randn((N, Dv))
    gamma = ones((1,)) / (2 * 0.01**2)  # kernel bandwidth
    alpha = ones((1,)) * 0.8  # regularization
    return x, b, gamma, alpha


######################################################################
# KeOps kernel
# ---------------
#
# Define a Gaussian RBF kernel:
#
formula = "Exp(- g * SqDist(x,y)) * a"
aliases = [
    "x = Vi(" + str(D) + ")",  # First arg:  i-variable of size D
    "y = Vj(" + str(D) + ")",  # Second arg: j-variable of size D
    "a = Vj(" + str(Dv) + ")",  # Third arg:  j-variable of size Dv
    "g = Pm(1)",
]  # Fourth arg: scalar parameter

######################################################################
# .. note::
#   This operator uses a conjugate gradient solver and assumes
#   that **formula** defines a **symmetric**, positive and definite
#   **linear** reduction with respect to the alias ``"a"``
#   specified trough the third argument.

######################################################################
# Define the Kernel solver, with a ridge regularization **alpha**:
#


def Kinv_keops(x, b, gamma, alpha, **kwargs):
    Kinv = KernelSolve(formula, aliases, "a", axis=1)
    res = Kinv(x, x, b, gamma, alpha=alpha)
    return res


def Kinv_keops_numpy(x, b, gamma, alpha, **kwargs):
    Kinv = KernelSolve_np(formula, aliases, "a", axis=1, dtype="float32")
    res = Kinv(x, x, b, gamma, alpha=alpha)
    return res


def Kinv_scipy(x, b, gamma, alpha, **kwargs):
    x_i = LazyTensor(np.sqrt(gamma) * x[:, None, :])
    y_j = LazyTensor(np.sqrt(gamma) * x[None, :, :])

    K_ij = (-((x_i - y_j) ** 2).sum(2)).exp()

    A = aslinearoperator(diags(alpha * np.ones(x.shape[0]))) + aslinearoperator(K_ij)
    A.dtype = np.dtype("float32")
    res = cg(A, b)
    return res


######################################################################
# Define the same Kernel solver, using a **tensorized** implementation:
#


def Kinv_pytorch(x, b, gamma, alpha, **kwargs):
    K_xx = alpha * torch.eye(x.shape[0], device=x.get_device()) + torch.exp(
        -gamma * sqdist_torch(x, x)
    )
    res = torchsolve(K_xx, b)
    return res


def Kinv_numpy(x, b, gamma, alpha, **kwargs):
    K_xx = alpha * np.eye(x.shape[0]) + np.exp(-gamma * sqdist_np(x, x))
    res = np.linalg.solve(K_xx, b)
    return res


######################################################################
# Run the benchmark
# ---------------------


routines = [
    (Kinv_numpy, "NumPy", {"lang": "numpy"}),
    (Kinv_pytorch, "PyTorch", {}),
    (Kinv_keops_numpy, "NumPy + KeOps", {"lang": "numpy"}),
    (Kinv_keops, "PyTorch + KeOps", {}),
    (Kinv_scipy, "Scipy + KeOps", {"lang": "numpy"}),
]
full_benchmark(
    "Inverse radial kernel matrix",
    routines,
    generate_samples,
    problem_sizes=problem_sizes,
    max_time=1,
)

plt.show()
