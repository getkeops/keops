"""
Inverse kernel operation
========================
"""

#####################################################################
# Setup
# --------------------
# Standard imports:

import os
import numpy as np
import time
from matplotlib import pyplot as plt

import importlib
import torch

from pykeops.numpy import KernelSolve as KernelSolve_np
from pykeops.torch import KernelSolve
from pykeops.torch.utils import squared_distances

use_cuda = torch.cuda.is_available() * 0
##############################################
# Benchmark specifications:
# 

D  = 3                        # Let's do this in 3D
Dv = 2                        # Dimension of the vectors (= number of linear problems to solve)
MAXTIME = 20 if use_cuda else 1   # Max number of seconds before we break the loop
REDTIME = 2  if use_cuda else .2  # Decrease the number of runs if computations take longer than 2s...

# Number of samples that we'll loop upon
NS = [10, 20, 50,
      100, 200, 500, 
      1000, 2000, 5000, 
      # 10000, 20000, 50000, 
      # 100000, 200000, 500000,
      # 1000000
      ]

######################################################################
# Create some random input data:
#

def generate_samples(N, device, lang):
    """Create point clouds sampled non-uniformly on a sphere of diameter 1."""
    if lang == 'torch':
        if device == 'cuda':
            torch.cuda.manual_seed_all(1234)
        else:
            torch.manual_seed(1234)

        x  = torch.rand(N, D, device=device)
        b  = torch.randn(N, Dv, device=device)
        sigma = torch.ones(1, device=device) * 1 / .01 ** 2   # kernel bandwidth
        alpha = torch.ones(1, device=device) * 0.8  # regularization
    else:
        np.random.seed(1234)

        x  = np.random.rand(N, D)
        b  = np.random.randn(N, Dv)
        sigma = np.ones(1) * 1 / .01 ** 2   # kernel bandwidth
        alpha = np.ones(1) * 0.8  # regularization

    return x, b, sigma, alpha

###############################################################################
# KeOps kernel
# ---------------
#
# Define a Gaussian RBF kernel:
#
formula = 'Exp(- g * SqDist(x,y)) * b'
aliases = ['x = Vx(' + str(D) + ')',   # First arg:  i-variable of size D
           'y = Vy(' + str(D) + ')',   # Second arg: j-variable of size D
           'b = Vy(' + str(Dv) + ')',  # Third arg:  j-variable of size Dv
           'g = Pm(1)']                # Fourth arg: scalar parameter

###############################################################################
# .. note::
#   This operator uses a conjugate gradient solver and assumes
#   that **formula** defines a **symmetric**, positive and definite
#   **linear** reduction with respect to the alias ``"b"``
#   specified trough the third argument.

###############################################################################
# Define the inverse kernel operation, with a ridge regularization **alpha**:
# 

def Kinv_keops(x, b, sigma, alpha):
    Kinv = KernelSolve(formula, aliases, "b", alpha=alpha, axis=1)
    res = Kinv(x, x, b, sigma)
    return res

def Kinv_keops_numpy(x, b, sigma, alpha):
    Kinv = KernelSolve_np(formula, aliases, "b", alpha=alpha, axis=1)
    res = Kinv(x, x, b, sigma)
    return res
###############################################################################
# Define the same  inverse kernel operation, using a **tensorized** implementation:
#
def Kinv_pytorch(x, b, sigma, alpha):
    K_xx = alpha * torch.eye(x.shape[0], device=x.get_device()) + torch.exp( - squared_distances(x, x) * sigma) 
    res = torch.gesv(b, K_xx)[0]
    return res

###############################################################################
# Define the same  inverse kernel operation, using a **tensorized** implementation:
#
def Kinv_numpy(x, b, sigma, alpha):
    K_xx = alpha * np.eye(x.shape[0]) + np.exp( - sigma * np.sum( (x[:,None,:] - x[None,:,:]) **2, axis=2) )
    res = np.linalg.solve(K_xx, b)
    return res

##############################################
# Benchmarking loops
# -----------------------

def benchmark(Routine, dev, N, loops=10, lang='torch') :
    """Times a convolution on an N-by-N problem."""

    importlib.reload(torch)  # In case we had a memory overflow just before...
    device = torch.device(dev)
    x, b, sigma, alpha = generate_samples(N, device, lang)

    # We simply benchmark a convolution
    code = "a = Routine(x, b, sigma, alpha)"
    exec( code, locals() ) # Warmup run, to compile and load everything

    t_0 = time.perf_counter()  # Actual benchmark --------------------
    for i in range(loops):
        if use_cuda: torch.cuda.synchronize()
        exec( code, locals() )
        if use_cuda: torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_0  # ---------------------------

    print("{:3} NxN convolution, with N ={:7}: {:3}x{:3.6f}s".format(loops, N, loops, elapsed / loops))
    return elapsed / loops


def bench_config(Routine, backend, dev, l) :
    """Times a convolution for an increasing number of samples."""

    print("Backend : {}, Device : {} -------------".format(backend, dev))

    times = []
    try :
        Nloops = [10, 1]
        nloops = Nloops.pop(0)
        for n in NS :
            elapsed = benchmark(Routine, dev, n, loops=nloops, lang=l)

            times.append( elapsed )
            if (nloops * elapsed > MAXTIME) \
            or (nloops * elapsed > REDTIME/10 and len(Nloops) > 0 ) : 
                nloops = Nloops.pop(0)

    except RuntimeError :
        print("**\nMemory overflow !")
    except IndexError :
        print("**\nToo slow !")
    
    return times + (len(NS)-len(times)) * [np.nan]


def full_bench(title, routines) :
    """Benchmarks the varied backends of a geometric loss function."""

    backends = [ backend for (_, backend, _) in routines ]

    print("Benchmarking : {} ===============================".format(title))
    
    lines  = [ NS ]
    for routine, backend, lang in routines :
        lines.append( bench_config(routine, backend, "cuda" if use_cuda else "cpu", lang) )

    benches = np.array(lines).T

    # Creates a pyplot figure:
    plt.figure(figsize=(12,8))
    linestyles = ["o-", "s-", "^-", "x-"]
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

    plt.title('Runtimes for {} in dimension {}'.format(title, D))
    plt.xlabel('Number of samples')
    plt.ylabel('Seconds')
    plt.yscale('log') ; plt.xscale('log')
    plt.legend(loc='upper left')
    plt.grid(True, which="major", linestyle="-")
    plt.grid(True, which="minor", linestyle="dotted")
    plt.axis([ NS[0], NS[-1], 1e-4, MAXTIME ])
    plt.tight_layout()

    # Save as a .csv to put a nice Tikz figure in the papers:
    header = "Npoints " + " ".join(backends)
    os.makedirs("output", exist_ok=True)
    np.savetxt("output/benchmark_convolutions_3D.csv", benches, 
               fmt='%-9.5f', header=header, comments='')


##############################################
# Run the benchmark
# ---------------------

routines = [ (Kinv_keops,   "KeOps_torch", "torch"),
             (Kinv_pytorch, "PyTorch", "torch"),  
             (Kinv_keops_numpy, "KeOps_numpy", "numpy"),  
             (Kinv_numpy, "Numpy", "numpy"),  
           ]
full_bench( "Kernel inverse operation", routines )

plt.show()
