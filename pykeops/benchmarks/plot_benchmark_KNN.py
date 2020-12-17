"""
K-Nearest Neighbours search (WIP)
===========================================================

Let's compare the performances of PyTorch, JAX, FAISS and KeOps fpr 
K-NN queries on random samples and standard datasets.

.. note::
    In this demo, we use exact **bruteforce** computations 
    (tensorized for PyTorch and online for KeOps), without leveraging any multiscale
    or low-rank (Nystroem/multipole) decomposition of the Kernel matrix.
    First support for these approximation schemes is scheduled for
    May-June 2021.

 
"""


##############################################
# Setup
# ---------------------

import numpy as np
import torch
from matplotlib import pyplot as plt
from functools import partial

from benchmark_utils import (
    flatten,
    random_normal,
    full_benchmark,
    timer,
    tensor,
    int_tensor,
    jax_tensor,
)
from dataset_utils import generate_samples

use_cuda = torch.cuda.is_available()


##############################################
# Benchmark specifications:
#

# Values of K that we'll loop upon:
Ks = [1, 2, 5, 10, 20, 50, 100]


##############################################
# Simple bruteforce implementations
# ------------------------------------------
#
# Define a simple Gaussian RBF product, using a **tensorized** implementation.
# Note that expanding the squared norm :math:`\|x-y\|^2` as a sum
# :math:`\|x\|^2 - 2 \langle x, y \rangle + \|y\|^2` allows us
# to leverage the fast matrix-matrix product of the BLAS/cuBLAS
# libraries.
#
#
# PyTorch bruteforce:
#

"""
def KNN_KeOps(K, metric="euclidean", **kwargs):
    def fit(x_train):
        # Setup the K-NN estimator:
        x_train = tensor(x_train)
        start = timer()

        # N.B.: The "training" time here should be negligible.
        elapsed = timer() - start

        def f(x_test):
            x_test = tensor(x_test)
            start = timer()

            # Actual K-NN query:

            elapsed = timer() - start

            indices = indices.cpu().numpy()
            return indices, elapsed

        return f, elapsed

    return fit
"""


def KNN_torch(K, metric="euclidean", **kwargs):
    def fit(x_train):
        # Setup the K-NN estimator:
        x_train = tensor(x_train)
        start = timer()
        # The "training" time here should be negligible:
        x_train_norm = (x_train ** 2).sum(-1)
        elapsed = timer() - start

        def f(x_test):
            x_test = tensor(x_test)
            start = timer()

            # Actual K-NN query:
            if metric == "euclidean":
                x_test_norm = (x_test ** 2).sum(-1)
                diss = (
                    x_test_norm.view(-1, 1)
                    + x_train_norm.view(1, -1)
                    - 2 * x_test @ x_train.t()
                )

            elif metric == "manhattan":
                diss = (x_test[:, None, :] - x_train[None, :, :]).abs().sum(dim=2)

            elif metric == "angular":
                diss = -x_test @ x_train.t()

            elif metric == "hyperbolic":
                x_test_norm = (x_test ** 2).sum(-1)
                diss = (
                    x_test_norm.view(-1, 1)
                    + x_train_norm.view(1, -1)
                    - 2 * x_test @ x_train.t()
                )
                diss /= x_test[:, 0].view(-1, 1) * x_train[:, 0].view(1, -1)

            out = diss.topk(K, dim=1, largest=False)

            elapsed = timer() - start
            indices = out.indices.cpu().numpy()
            return indices, elapsed

        return f, elapsed

    return fit


#############################################################################
# PyTorch bruteforce, with small batches to avoid memory overflows:


def KNN_torch_batch_loop(K, metric="euclidean", **kwargs):
    def fit(x_train):
        # Setup the K-NN estimator:
        x_train = tensor(x_train)
        Ntrain, D = x_train.shape
        start = timer()
        # The "training" time here should be negligible:
        x_train_norm = (x_train ** 2).sum(-1)
        elapsed = timer() - start

        def f(x_test):
            x_test = tensor(x_test)

            # Estimate the largest reasonable batch size:
            Ntest = x_test.shape[0]
            #  torch.cuda.get_device_properties(deviceId).total_memory
            av_mem = int(5e8)
            Ntest_loop = min(max(1, av_mem // (4 * D * Ntrain)), Ntest)
            Nloop = (Ntest - 1) // Ntest_loop + 1
            # print(f"{Ntest} queries, split in {Nloop} batches of {Ntest_loop} queries each.")
            out = int_tensor(Ntest, K)

            start = timer()
            # Actual K-NN query:
            for k in range(Nloop):
                x_test_k = x_test[Ntest_loop * k : Ntest_loop * (k + 1), :]
                if metric == "euclidean":
                    x_test_norm = (x_test_k ** 2).sum(-1)
                    diss = (
                        x_test_norm.view(-1, 1)
                        + x_train_norm.view(1, -1)
                        - 2 * x_test_k @ x_train.t()
                    )

                elif metric == "manhattan":
                    diss = (x_test_k[:, None, :] - x_train[None, :, :]).abs().sum(dim=2)

                elif metric == "angular":
                    diss = -x_test_k @ x_train.t()

                elif metric == "hyperbolic":
                    x_test_norm = (x_test_k ** 2).sum(-1)
                    diss = (
                        x_test_norm.view(-1, 1)
                        + x_train_norm.view(1, -1)
                        - 2 * x_test_k @ x_train.t()
                    )
                    diss /= x_test_k[:, 0].view(-1, 1) * x_train[:, 0].view(1, -1)

                out[Ntest_loop * k : Ntest_loop * (k + 1), :] = diss.topk(
                    K, dim=1, largest=False
                ).indices
                del diss
            # torch.cuda.empty_cache()

            elapsed = timer() - start
            indices = out.cpu().numpy()
            return indices, elapsed

        return f, elapsed

    return fit


############################################################################
# Distance matrices with JAX:

from functools import partial
import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(2, 3))
def knn_jax_fun(x_train, x_test, K, metric):
    if metric == "euclidean":
        diss = (
            (x_test ** 2).sum(-1)[:, None]
            + (x_train ** 2).sum(-1)[None, :]
            - 2 * x_test @ x_train.T
        )
    elif metric == "manhattan":
        diss = jax.lax.abs(x_test[:, None, :] - x_train[None, :, :]).sum(-1)
    elif metric == "angular":
        diss = -x_test @ x_train.T
    elif metric == "hyperbolic":
        diss = (
            (x_test ** 2).sum(-1)[:, None]
            + (x_train ** 2).sum(-1)[None, :]
            - 2 * x_test @ x_train.T
        )
        diss = diss / (x_test[:, 0][:, None] * x_train[:, 0][None, :])

    indices = jax.lax.top_k(-diss, K)[1]
    return indices


############################################################################
# JAX bruteforce:


def KNN_JAX(K, metric="euclidean", **kwargs):
    def fit(x_train):

        # Setup the K-NN estimator:
        start = timer(use_torch=False)
        x_train = jax_tensor(x_train)
        elapsed = timer(use_torch=False) - start

        def f(x_test):
            x_test = jax_tensor(x_test)

            # Actual K-NN query:
            start = timer(use_torch=False)
            indices = knn_jax_fun(x_train, x_test, K, metric)
            indices = np.array(indices)
            elapsed = timer(use_torch=False) - start
            return indices, elapsed

        return f, elapsed

    return fit


#############################################################################
# JAX bruteforce, with small batches to avoid memory overflows:


def KNN_JAX_batch_loop(K, metric="euclidean", **kwargs):
    def fit(x_train):

        # Setup the K-NN estimator:
        start = timer(use_torch=False)
        x_train = jax_tensor(x_train)
        elapsed = timer(use_torch=False) - start

        def f(x_test):
            x_test = jax_tensor(x_test)

            # Estimate the largest reasonable batch size
            #  torch.cuda.get_device_properties(deviceId).total_memory
            av_mem = int(5e8)
            Ntrain, D = x_train.shape
            Ntest = x_test.shape[0]
            Ntest_loop = min(max(1, av_mem // (4 * D * Ntrain)), Ntest)
            Nloop = (Ntest - 1) // Ntest_loop + 1
            # print(f"{Ntest} queries, split in {Nloop} batches of {Ntest_loop} queries each.")
            indices = np.zeros((Ntest, K), dtype=int)

            start = timer(use_torch=False)
            # Actual K-NN query:
            for k in range(Nloop):
                x_test_k = x_test[Ntest_loop * k : Ntest_loop * (k + 1), :]
                indices[Ntest_loop * k : Ntest_loop * (k + 1), :] = knn_jax_fun(
                    x_train, x_test_k, K, metric
                )
            elapsed = timer(use_torch=False) - start
            return indices, elapsed

        return f, elapsed

    return fit


############################################################################
# KeOps bruteforce implementation:
#

from pykeops.torch import LazyTensor, Vi, Vj


def KNN_KeOps(K, metric="euclidean", **kwargs):
    def fit(x_train):
        # Setup the K-NN estimator:
        x_train = tensor(x_train)
        start = timer()

        # Encoding as KeOps LazyTensors:
        D = x_train.shape[1]
        X_i = Vi(0, D)
        X_j = Vj(1, D)

        # Symbolic distance matrix:
        if metric == "euclidean":
            D_ij = ((X_i - X_j) ** 2).sum(-1)
        elif metric == "manhattan":
            D_ij = (X_i - X_j).abs().sum(-1)
        elif metric == "angular":
            D_ij = -(X_i | X_j)
        elif metric == "hyperbolic":
            D_ij = ((X_i - X_j) ** 2).sum(-1) / (X_i[0] * X_j[0])

        # K-NN query operator:
        KNN_fun = D_ij.argKmin(K, dim=1)

        # N.B.: The "training" time here should be negligible.
        elapsed = timer() - start

        def f(x_test):
            x_test = tensor(x_test)
            start = timer()

            # Actual K-NN query:
            indices = KNN_fun(x_test, x_train)

            elapsed = timer() - start

            indices = indices.cpu().numpy()
            return indices, elapsed

        return f, elapsed

    return fit


################################################################################
# SciKit-Learn tree-based and bruteforce methods
# -----------------------------------------------------
#


from sklearn.neighbors import NearestNeighbors


def KNN_sklearn(K, metric="euclidean", algorithm=None, **kwargs):

    if metric in ["euclidean", "angular"]:
        p = 2
    elif metric == "manhattan":
        p = 1
    else:
        raise NotImplementedError("This distance is not supported.")

    KNN_meth = NearestNeighbors(n_neighbors=K, algorithm=algorithm, p=p, n_jobs=-1)

    def fit(x_train):
        # Setup the K-NN estimator:
        start = timer()
        KNN_fun = KNN_meth.fit(x_train).kneighbors
        elapsed = timer() - start

        def f(x_test):
            start = timer()
            distances, indices = KNN_fun(x_test)
            elapsed = timer() - start

            return indices, elapsed

        return f, elapsed

    return fit


KNN_sklearn_auto = partial(KNN_sklearn, algorithm="auto")
KNN_sklearn_ball_tree = partial(KNN_sklearn, algorithm="ball_tree")
KNN_sklearn_kd_tree = partial(KNN_sklearn, algorithm="kd_tree")
KNN_sklearn_brute = partial(KNN_sklearn, algorithm="brute")


##############################################
# NumPy vs. PyTorch vs. KeOps (Gpu)
# --------------------------------------------------------


def run_KNN_benchmark(name):

    # Load the dataset and some info:
    dataset = generate_samples(name)(1)
    N_train, dimension = dataset["train"].shape
    N_test, _ = dataset["test"].shape
    metric = dataset["metric"]

    # Routines to benchmark:
    routines = [
        # (KNN_sklearn_auto, "sklearn, auto (CPU)", {}),
        # (KNN_sklearn_ball_tree, "sklearn, Ball-tree (CPU)", {}),
        # (KNN_sklearn_kd_tree, "sklearn, KD-tree (CPU)", {}),
        # (KNN_sklearn_brute, "sklearn, bruteforce (CPU)", {}),
        (KNN_torch, "PyTorch (GPU)", {}),
        (KNN_torch_batch_loop, "PyTorch (small batches, GPU)", {}),
        (KNN_KeOps, "KeOps (GPU)", {}),
        (KNN_JAX, "JAX (GPU)", {}),
        (KNN_JAX_batch_loop, "JAX (small batches, GPU)", {}),
    ]

    # Actual run:
    full_benchmark(
        f"K-NN search on {name}: {N_test:,} queries on a dataset of {N_train:,} points\nin dimension {dimension:,} with a {metric} metric.",
        routines,
        generate_samples(name),
        min_time=1e-4,
        max_time=10,
        problem_sizes=Ks,
        xlabel="Number of neighbours K",
    )


##############################################
# On random samples:
# --------------------------------------------------------
#
# Small dataset in :math:`\mathbb{R}^3`:

run_KNN_benchmark("R^D a")

########################################
# Large dataset in :math:`\mathbb{R}^3`:

run_KNN_benchmark("R^D b")

plt.show()
