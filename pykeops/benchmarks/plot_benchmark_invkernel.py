"""
Solving positive definite linear systems
=========================================

This benchmark compares the performances of KeOps versus Numpy and Pytorch on a inverse matrix operation. It uses the functions :class:`torch.KernelSolve <pykeops.torch.KernelSolve>` (see also :doc:`here <../_auto_examples/pytorch/plot_test_invkernel_torch>`) and  :class:`numpy.KernelSolve <pykeops.numpy.KernelSolve>` (see also :doc:`here <../_auto_examples/numpy/plot_test_invkernel_numpy>`).
 
In a nutshell, given :math:`x \in\mathbb R^{N\\times D}`  and :math:`b \in \mathbb R^{N\\times D_v}`, we compute :math:`a \in \mathbb R^{N\\times D_v}` so that

.. math::

  b = (\\alpha\operatorname{Id} + K_{x,x}) a \quad \Leftrightarrow \quad a = (\\alpha\operatorname{Id}+ K_{x,x})^{-1} b
  
where :math:`K_{x,x} = \Big[\exp(-\|x_i -x_j\|^2 / \sigma^2)\Big]_{i,j=1}^N`. The method is based on a conjugate gradient scheme. The benchmark tests various values of :math:`N \in [10, \cdots,10^6]`.

 
"""

#####################################################################
# Setup
# -----
# Standard imports:

import importlib
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import aslinearoperator, cg

from pykeops.numpy import KernelSolve as KernelSolve_np, LazyTensor
from pykeops.torch import KernelSolve
from pykeops.torch.utils import squared_distances
from pykeops.numpy import Vi, Vj, Pm


use_cuda = torch.cuda.is_available()

#####################################################################
# Benchmark specifications:
# 

D  = 3  # Let's do this in 3D
Dv = 1  # Dimension of the vectors (= number of linear problems to solve)
MAXTIME = 10 if use_cuda else 1   # Max number of seconds before we break the loop
REDTIME = 5  if use_cuda else .2  # Decrease the number of runs if computations take longer than 2s...

# Number of samples that we'll loop upon
NS = [10, 20, 50,
      100, 200, 500, 
      1000, 2000, 5000, 
      10000, 20000, 50000, 
      100000, 200000, 500000,
      1000000
      ]

#####################################################################
# Create some random input data:
#

def generate_samples(N, device, lang):
    """Create point clouds sampled non-uniformly on a sphere of diameter 1."""
    if lang == 'torch':
        if device == 'cuda':
            torch.cuda.manual_seed_all(1234)
        else:
            torch.manual_seed(1234)

        x = torch.rand(N, D, device=device)
        b = torch.randn(N, Dv, device=device)
        gamma = torch.ones(1, device=device) * .5 / .01 ** 2  # kernel bandwidth
        alpha = torch.ones(1, device=device) * 2  # regularization
    else:
        np.random.seed(1234)

        x  = np.random.rand(N, D).astype('float32')
        b  = np.random.randn(N, Dv).astype('float32')
        gamma = (np.ones(1) * 1 / .01 ** 2).astype('float32')   # kernel bandwidth
        alpha = (np.ones(1) * 0.8).astype('float32')  # regularization

    return x, b, gamma, alpha

######################################################################
# KeOps kernel
# ---------------
#
# Define a Gaussian RBF kernel:
#
formula = 'Exp(- g * SqDist(x,y)) * a'
aliases = ['x = Vi(' + str(D) + ')',   # First arg:  i-variable of size D
           'y = Vj(' + str(D) + ')',   # Second arg: j-variable of size D
           'a = Vj(' + str(Dv) + ')',  # Third arg:  j-variable of size Dv
           'g = Pm(1)']                # Fourth arg: scalar parameter

######################################################################
# .. note::
#   This operator uses a conjugate gradient solver and assumes
#   that **formula** defines a **symmetric**, positive and definite
#   **linear** reduction with respect to the alias ``"a"``
#   specified trough the third argument.

######################################################################
# Define the Kernel solver, with a ridge regularization **alpha**:
# 

def Kinv_keops(x, b, gamma, alpha):
    Kinv = KernelSolve(formula, aliases, "a", axis=1)
    res = Kinv(x, x, b, gamma, alpha=alpha)
    return res

def Kinv_keops_numpy(x, b, gamma, alpha):
    Kinv = KernelSolve_np(formula, aliases, "a", axis=1, dtype='float32')
    res = Kinv(x, x, b, gamma, alpha=alpha)
    return res

def Kinv_scipy(x, b, gamma, alpha):
    K_ij = (-Pm(gamma) * Vi(x).sqdist(Vj(x))).exp()
    A = aslinearoperator(
        diags(alpha * np.ones(x.shape[0]))) + aslinearoperator(K_ij)
    A.dtype = np.dtype('float32')
    res = cg(A, b)
    return res


######################################################################
# Define the same Kernel solver, using a **tensorized** implementation:
#

def Kinv_pytorch(x, b, gamma, alpha):
    K_xx = alpha * torch.eye(x.shape[0], device=x.get_device()) + torch.exp( - squared_distances(x, x) * gamma)
    res = torch.solve(b, K_xx)[0]
    return res

def Kinv_numpy(x, b, gamma, alpha):
    K_xx = alpha * np.eye(x.shape[0]) + np.exp( - gamma * np.sum( (x[:,None,:] - x[None,:,:]) **2, axis=2) )
    res = np.linalg.solve(K_xx, b)
    return res

######################################################################
# Benchmarking loops
# -----------------------

def benchmark(Routine, dev, N, loops=10, lang='torch') :
    """Times a routine on an N-by-N problem."""

    importlib.reload(torch)  # In case we had a memory overflow just before...
    device = torch.device(dev)
    x, b, gamma, alpha = generate_samples(N, device, lang)

    # We simply benchmark a kernel inversion
    code = "a = Routine(x, b, gamma, alpha)"
    exec( code, locals() ) # Warmup run, to compile and load everything
    if use_cuda: torch.cuda.synchronize()

    t_0 = time.perf_counter()  # Actual benchmark --------------------
    for i in range(loops):
        exec( code, locals() )
    if use_cuda: torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_0  # ---------------------------

    print("{:3} NxN kernel inversion, with N ={:7}: {:3}x{:3.6f}s".format(loops, N, loops, elapsed / loops))
    return elapsed / loops


def bench_config(Routine, backend, dev, l) :
    """Times a routine for an increasing number of samples."""

    print("Backend : {}, Device : {} -------------".format(backend, dev))

    times = []
    not_recorded_times = []
    try :
        Nloops = [100, 10, 1]
        nloops = Nloops.pop(0)
        for n in NS :
            elapsed = benchmark(Routine, dev, n, loops=nloops, lang=l)

            times.append( elapsed )
            if (nloops * elapsed > MAXTIME) or (nloops * elapsed > REDTIME/nloops and len(Nloops) > 0): 
                nloops = Nloops.pop(0)

    except RuntimeError:
        print("**\nMemory overflow !")
        not_recorded_times = (len(NS)-len(times)) * [np.nan]
    except IndexError:
        print("**\nToo slow !")
        not_recorded_times = (len(NS)-len(times)) * [np.Infinity]
    
    return times + not_recorded_times


def full_bench(title, routines) :
    """Benchmarks a collection of routines."""

    backends = [ backend for (_, backend, _) in routines ]

    print("Benchmarking : {} ===============================".format(title))
    
    lines  = [ NS ]
    for routine, backend, lang in routines :
        lines.append(bench_config(routine, backend, "cuda" if use_cuda else "cpu", lang) )

    benches = np.array(lines).T

    # Creates a pyplot figure:
    plt.figure(figsize=(12,8))
    linestyles = ["o-", "s-", "^-", "x-", "<-"]
    for i, backend in enumerate(backends):
        plt.plot( benches[:,0], benches[:,i+1], linestyles[i], 
                  linewidth=2, label='backend = "{}"'.format(backend) )
        
        for (j, val) in enumerate( benches[:,i+1] ):
            if np.isnan(val) and j > 0:
                x, y = benches[j-1,0], benches[j-1,i+1]
                plt.annotate('Memory overflow!',
                    xy=(x, 1.05*y),
                    horizontalalignment='center',
                    verticalalignment='bottom')
                break
            elif np.isinf(val) and j > 0:
                x, y = benches[j-1,0], benches[j-1,i+1]
                plt.annotate('Too slow!',
                    xy=(x, 1.05*y),
                    horizontalalignment='center',
                    verticalalignment='bottom')
                break

    plt.title('Runtimes for {} in dimension {}'.format(title, D))
    plt.xlabel('Number of samples')
    plt.ylabel('Seconds')
    plt.yscale('log') ; plt.xscale('log')
    plt.legend(loc='upper left')
    plt.grid(True, which="major", linestyle="-")
    plt.grid(True, which="minor", linestyle="dotted")
    plt.tight_layout()

    # Save as a .csv to put a nice Tikz figure in the papers:
    header = "Npoints " + " ".join(backends)
    os.makedirs("output", exist_ok=True)
    np.savetxt("output/benchmark_kernelsolve.csv", benches, 
               fmt='%-9.5f', header=header, comments='')


######################################################################
# Run the benchmark
# ---------------------

routines = [(Kinv_numpy, "NumPy", "numpy"), 
            (Kinv_pytorch, "PyTorch", "torch"),  
            (Kinv_keops_numpy, "NumPy + KeOps", "numpy"),  
            (Kinv_keops,   "PyTorch + KeOps", "torch"),
            (Kinv_scipy,   "Scipy + KeOps", "numpy"),
           ]
full_bench( "Inverse radial kernel matrix", routines )

plt.show()
