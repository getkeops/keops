"""
Mixed-precision and accuracy settings
===========================================================

We test various options of KeOps regarding accuracy of computations.
 
"""


##############################################
# Setup
# ---------------------

output_filename = "accuracy"

import importlib
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

use_cuda = torch.cuda.is_available()

D = 3

##############################################
# Benchmark specifications:
#

MAXTIME = 10 if use_cuda else 1  # Max number of seconds before we break the loop
REDTIME = (
    2 if use_cuda else 0.2
)  # Decrease the number of runs if computations take longer than 2s...

# Number of samples that we'll loop upon
NS = [
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
    200000,
    500000,
    1000000,
    2000000,
    5000000,
]

##############################################
# Synthetic dataset.


def generate_samples(N, D, device, lang):
    """Create point clouds sampled non-uniformly on a sphere of diameter 1."""

    if lang == "torch":
        if device == "cuda":
            torch.cuda.manual_seed_all(123)
        else:
            torch.manual_seed(123)

        x = torch.rand((N, D), device=device, dtype=torch.float64)
        y = torch.rand((N, D), device=device, dtype=torch.float64)
        # Draw a random source signal:
        b = torch.randn((N, 1), device=device, dtype=torch.float64)

    else:
        np.random.seed(1234)

        x = np.random.randn(*((N, D)))
        y = np.random.randn(*((N, D)))
        b = np.random.randn(*((N,)))

    return x, y, b


##############################################
# Define a simple RBF product, using the :class:`pykeops.torch.LazyTensor` wrapper:

from pykeops.torch import LazyTensor


def conv_lazytensor(x, y, b, dtype, dtype_acc, sum_scheme):
    backend = "GPU" if use_cuda else "CPU"
    x_i = LazyTensor(x.unsqueeze(-2))  # (M, 1, D)
    y_j = LazyTensor(y.unsqueeze(-3))  # (1, N, D)
    K_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N, 1)
    S_ij = K_ij * b.unsqueeze(-3)  # (M, N, 1) * (1, N, 1)
    return S_ij.sum(dim=1, backend=backend, dtype_acc=dtype_acc, sum_scheme=sum_scheme)


##############################################
# Benchmarking loops
# -----------------------


def benchmark(Routine, dev, N, D, loops, lang, dtype, dtype_acc, sum_scheme):
    """Times a convolution on an N-by-N problem, and evaluate accuracy."""

    importlib.reload(torch)  # In case we had a memory overflow just before...
    device = torch.device(dev)
    x_, y_, b_ = generate_samples(N, D, device, lang)
    if dtype == "float16":
        torch_dtype = torch.float16
    if dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "float64":
        torch_dtype = torch.float64
    x, y, b = x_.to(torch_dtype), y_.to(torch_dtype), b_.to(torch_dtype)

    # We simply benchmark a convolution

    N0 = min(N, 100)
    Routine(
        x[:N0, :], y[:N0, :], b[:N0, :], dtype, dtype_acc, sum_scheme
    )  # Warmup run, to compile and load everything

    # timings
    if loops > 0:
        code = "out = Routine( x, y, b, dtype, dtype_acc, sum_scheme ) "
        t_0 = time.perf_counter()  # Actual benchmark --------------------
        if use_cuda:
            torch.cuda.synchronize()
        for i in range(loops):
            exec(code, locals())
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_0  # ---------------------------
        elapsed /= loops
        print(
            "timing of {:3} NxN convolution(s), with N ={:7}: {:3}x{:3.6f}s".format(
                loops, N, loops, elapsed / loops
            )
        )
    else:
        elapsed = np.NaN

    # accuracy
    ind = torch.randperm(y.shape[0])
    M = min(
        N, 1000
    )  # we evaluate accuracy on a subsample of outputs only because computations with full precisions are slow.
    out = Routine(x[:M, :], y[ind, :], b[ind, :], dtype, dtype_acc, sum_scheme)
    ref_out = Routine(x_[:M, :], y_, b_, "float64", "float64", "kahan_scheme")
    mean_err = (
        (out.double() - ref_out.double()).abs().mean() / ref_out.double().abs().mean()
    ).item()
    mean_err = float("NaN") if mean_err == 0 else mean_err
    max_err = (
        (out.double() - ref_out.double()).abs().max() / ref_out.double().abs().mean()
    ).item()
    max_err = float("NaN") if max_err == 0 else max_err
    print(
        "accuracy of an MxN convolution, with M = {}, N ={:7}: mean err={:.1e}, max err={:.1e}".format(
            M, N, mean_err, max_err
        )
    )

    return elapsed, mean_err, max_err


def bench_config(Routine, backend, dev, lang, dtype, dtype_acc, sum_scheme):
    """Times a convolution for an increasing number of samples."""

    print(
        "Backend : {}, Device : {}, dtype : {}, dtype_acc : {}, sum_scheme : {} -------------".format(
            backend, dev, dtype, dtype_acc, sum_scheme
        )
    )

    times = []
    mean_errs = []
    max_errs = []

    try:
        Nloops = [100, 10, 1, 0]
        nloops = Nloops.pop(0)
        for n in NS:
            elapsed, mean_err, max_err = benchmark(
                Routine, dev, n, D, nloops, lang, dtype, dtype_acc, sum_scheme
            )
            times.append(elapsed)
            mean_errs.append(mean_err)
            max_errs.append(max_err)
            if nloops > 0:
                if (nloops * elapsed > MAXTIME) or (
                    nloops * elapsed > REDTIME / 10 and nloops > 1
                ):
                    nloops = Nloops.pop(0)

    except RuntimeError:
        print("**\nMemory overflow !")
    except IndexError:
        print("**\nToo slow !")

    fill_nans = (len(NS) - len(times)) * [np.nan]
    return times + fill_nans, mean_errs + fill_nans, max_errs + fill_nans


def full_bench(title, routines):
    """Benchmarks the varied options of a geometric loss function."""

    backends = [backend for (_, backend, _, _, _, _) in routines]

    print("Benchmarking : {} ===============================".format(title))

    lines_times = [NS]
    lines_mean_errs = [NS]
    lines_max_errs = [NS]
    for routine, backend, lang, dtype, dtype_acc, sum_scheme in routines:
        res = bench_config(
            routine,
            backend,
            "cuda" if use_cuda else "cpu",
            lang,
            dtype,
            dtype_acc,
            sum_scheme,
        )
        lines_times.append(res[0])
        lines_mean_errs.append(res[1])
        lines_max_errs.append(res[2])

    benches_times = np.array(lines_times).T
    benches_mean_errs = np.array(lines_mean_errs).T
    benches_max_errs = np.array(lines_max_errs).T

    for ind_benches, benches in enumerate(
        (benches_times, benches_mean_errs, benches_max_errs)
    ):
        # Creates a pyplot figure:
        plt.figure(figsize=(12, 8))
        linestyles = ["o-", "s-", "^-", "<-", ">-", "v-", "+-", "*-", "x-", "p-", "d-"]
        for i, config in enumerate(routines):
            plt.plot(
                benches[:, 0],
                benches[:, i + 1],
                linestyles[i],
                linewidth=2,
                label='config = "{}"'.format(config[3:]),
            )

        plt.xlabel("Number of samples")
        if ind_benches == 0:
            plt.title("Runtimes for {} in dimension {}".format(title, D))
            plt.ylabel("Seconds")
        elif ind_benches == 1:
            plt.title("Mean errors for {} in dimension {}".format(title, D))
            plt.ylabel("Relative mean error")
        elif ind_benches == 2:
            plt.title("Max errors for {} in dimension {}".format(title, D))
            plt.ylabel("Relative max error")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend(loc="upper left")
        plt.grid(True, which="major", linestyle="-")
        plt.grid(True, which="minor", linestyle="dotted")
        true_vals = benches[:, 1:].flatten()
        true_vals = true_vals[np.isfinite(true_vals)]
        if ind_benches == 0:
            plt.axis([NS[0], NS[-1], true_vals.min(), MAXTIME])
        else:
            plt.axis([NS[0], NS[-1], true_vals.min(), 100 * true_vals.max()])
        plt.tight_layout()

        # Save as a .csv to put a nice Tikz figure in the papers:
        header = "Npoints " + " ".join(backends)
        os.makedirs("output", exist_ok=True)
        np.savetxt(
            "output/" + output_filename + "_" + str(ind_benches) + ".csv",
            benches,
            fmt="%-9.5f",
            header=header,
            comments="",
        )


##############################################
# KeOps
# --------------------------------------------------------

routines = [
    (
        conv_lazytensor,
        "float16, direct_sum",
        "torch",
        "float16",
        "float16",
        "direct_sum",
    ),
    (conv_lazytensor, "float16, block_sum", "torch", "float16", "float16", "block_sum"),
    (
        conv_lazytensor,
        "float16, kahan_scheme",
        "torch",
        "float16",
        "float16",
        "kahan_scheme",
    ),
    (
        conv_lazytensor,
        "float16, float32 acc",
        "torch",
        "float16",
        "float32",
        "block_sum",
    ),
    (
        conv_lazytensor,
        "float32, direct_sum",
        "torch",
        "float32",
        "float32",
        "direct_sum",
    ),
    (conv_lazytensor, "float32, block_sum", "torch", "float32", "float32", "block_sum"),
    (
        conv_lazytensor,
        "float32, kahan_scheme",
        "torch",
        "float32",
        "float32",
        "kahan_scheme",
    ),
    (
        conv_lazytensor,
        "float32, float64 acc",
        "torch",
        "float32",
        "float64",
        "block_sum",
    ),
    (
        conv_lazytensor,
        "float64, direct_sum",
        "torch",
        "float64",
        "float64",
        "direct_sum",
    ),
    (conv_lazytensor, "float64, block_sum", "torch", "float64", "float64", "block_sum"),
    (
        conv_lazytensor,
        "float64, kahan_scheme",
        "torch",
        "float64",
        "float64",
        "kahan_scheme",
    ),
]
full_bench(" Matrix-Vector products", routines)

plt.show()
