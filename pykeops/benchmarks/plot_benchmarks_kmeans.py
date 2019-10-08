r"""
Scaling up K-Means classification
===========================================================

Let's compare the performances of Numpy, PyTorch and KeOps on 
simple K-Means iteration in dimension 100,
with a growing number of samples :math:`\mathrm{N}` and
:math:`\mathrm{K} = \lfloor \sqrt{\mathrm{N}} \rfloor` clusters.
 
"""


##############################################
# Setup
# ---------------------

import importlib
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

use_cuda = torch.cuda.is_available()

##############################################
# Benchmark specifications:
# 

D  = 10        # Let's do this in dimension 100

MAXTIME = 10 if use_cuda else 1   # Max number of seconds before we break the loop
REDTIME = 2  if use_cuda else .2  # Decrease the number of runs if computations take longer than 2s...

# Number of samples that we'll loop upon
NS = [100, 200, 500, 
      1000, 2000, 5000, 
      10000, 20000, 50000, 
      100000, 200000, 500000,
      1000000]


dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}


##############################################
# Synthetic dataset. 

def generate_samples(N, device, lang, batchsize=None):
    """Create point clouds sampled non-uniformly on a sphere of diameter 1."""

    B = () if batchsize is None else (batchsize,)

    if lang == 'torch':
        if device == 'cuda':
            torch.cuda.manual_seed_all(1234)
        else:
            torch.manual_seed(1234)

        x = torch.rand(B + (N, D), device=device)

    else:
        np.random.seed(1234)
        x = np.random.rand(*(B + (N, D))).astype(dtype)

    return x


##############################################
# Define a simple K-nearest neighbors search, using a **tensorized** implementation:
#



def kmeans_numpy(x, K=10, Niter=10):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud, 
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    c = x[:K, :].copy()   # Simplistic random initialization
    x_i = x[:, None, :]   # (Npoints, 1, D)

    for i in range(Niter):
    
        c_j = c[None, :, :]  # (1, Nclusters, D)

        D_xx = (x*x).sum(-1)[:,None]         # (N,1)
        D_xc = x @ c.T  # (N,D) @ (D,K) = (N,K)
        D_cc = (c*c).sum(-1)[None,:]         # (1,K)
        D_xc = D_xx - 2*D_xc + D_cc

        cl = D_xc.argmin(axis=1).ravel()  # Points -> Nearest cluster

        Ncl = np.bincount(cl).astype(dtype)  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = np.bincount(cl, weights=x[:, d]) / Ncl

    return cl



def kmeans_pytorch(x, K=10, Niter=10):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud, 
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = x[:, None, :]   # (Npoints, 1, D)

    for i in range(Niter):
    
        c_j = c[None, :, :]  # (1, Nclusters, D)

        D_xx = (x*x).sum(-1).unsqueeze(1)         # (N,1)
        D_xc = torch.matmul( x, c.permute(1,0) )  # (N,D) @ (D,K) = (N,K)
        D_cc = (c*c).sum(-1).unsqueeze(0)         # (1,K)
        D_xc = D_xx - 2*D_xc + D_cc

        cl = D_xc.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    return cl



#############################################
# Finally, perform the same operation with our high-level :class:`pykeops.torch.LazyTensor` wrapper:

from pykeops.torch import LazyTensor


def kmeans_keops(x, K=10, Niter=10):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud, 
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    for i in range(Niter):
    
        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    return cl
        

##############################################
# Benchmarking loops
# -----------------------

def benchmark(routine_batchsize, dev, N, loops=10, lang='torch'):
    """Times a convolution on an N-by-N problem."""

    if isinstance(routine_batchsize, tuple):
        Routine, B = routine_batchsize
    else:
        Routine, B = routine_batchsize, None

    importlib.reload(torch)  # In case we had a memory overflow just before...
    device = torch.device(dev)
    x = generate_samples(N, device, lang, batchsize=B)
    K = int(np.floor(np.sqrt(N)))

    # We simply benchmark a convolution
    code = "a = Routine( x, K ) "
    exec( code, locals() ) # Warmup run, to compile and load everything

    t_0 = time.perf_counter()  # Actual benchmark --------------------
    if use_cuda: torch.cuda.synchronize()
    for i in range(loops):
        exec( code, locals() )
    if use_cuda: torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_0  # ---------------------------

    if B is None:
        print("{:3} 10 K-Means iteration, with N ={:7} and K = {:3}: {:3}x{:3.6f}s".format(loops, N, K, loops, elapsed / loops))
        return elapsed / loops
    else:
        print("{:3}x{:3} 10 K-Means iteration, with N ={:7} and K = {:3}: {:3}x{:3}x{:3.6f}s".format(
            B, loops, N, K, B, loops, elapsed / (B * loops)))
        return elapsed / (B * loops)


def bench_config(Routine, backend, dev, l) :
    """Times a convolution for an increasing number of samples."""

    print("Backend : {}, Device : {} -------------".format(backend, dev))

    times = []
    try :
        Nloops = [100, 10, 1]
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
    linestyles = ["o-", "s-", "^-"]
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
    plt.axis([NS[0], NS[-1], 5e-4, MAXTIME])
    plt.tight_layout()

    # Save as a .csv to put a nice Tikz figure in the papers:
    header = "Npoints " + " ".join(backends)
    os.makedirs("output", exist_ok=True)
    np.savetxt("output/benchmark_kmeans.csv", benches, 
               fmt='%-9.5f', header=header, comments='')


##############################################
# NumPy vs. PyTorch vs. KeOps
# --------------------------------------------------------

routines = [ (kmeans_numpy,   "NumPy",   "numpy"),
             (kmeans_pytorch, "PyTorch", "torch"),
             (kmeans_keops,   "KeOps",   "torch"), ]
full_bench( f"10 iterations of the K-Means loop", routines )


