"""
Utility functions for the benchmarks
==========================================

"""


import importlib
import os
import time
import matplotlib as mpl
from matplotlib import pyplot as plt
from si_prefix import si_format

import numpy as np
import torch

use_cuda = torch.cuda.is_available()


def mytime():
    if use_cuda:
        torch.cuda.synchronize()
    return time.perf_counter()


def flatten(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]


def random_normal(device="cuda", lang="torch"):
    def sampler(shape):
        if lang == "torch":
            return torch.randn(shape, device=torch.device(device))
        else:
            return np.random.rand(*shape).astype("float32")

    return sampler


def unit_tensor(device="cuda", lang="torch"):
    def sampler(shape):
        if lang == "torch":
            return torch.ones(shape, device=torch.device(device))
        else:
            return np.ones(*shape).astype("float32")

    return sampler


##############################################
# Benchmarking loops
# -----------------------


def benchmark(routine, label, N, loops=10, generate_samples=None, **kwargs):

    importlib.reload(torch)  # In case we had a memory overflow just before...
    args = generate_samples(N, **kwargs)

    # Warmup run, to compile and load everything:
    output = routine(*args, **kwargs)

    t_0 = mytime()  # Actual benchmark --------------------

    for i in range(loops):
        output = routine(*args, **kwargs)

    elapsed = mytime() - t_0  # ---------------------------

    B = kwargs.get("batchsize", 1)
    perf = elapsed / (B * loops)

    print(f"{B:3}x{loops:3} loops of size N ={N:9,}: {B:3}x{loops:3}x{perf:3.6f}s")
    return perf


def bench_config(
    routine,
    label,
    kwargs,
    generate_samples=None,
    problem_sizes=[1],
    max_time=10,
    red_time=2,
):
    """Times a convolution for an increasing number of samples."""

    print(f"{label} -------------")

    times = []
    not_recorded_times = []
    try:
        Nloops = [100, 10, 1]
        nloops = Nloops.pop(0)
        for n in problem_sizes:
            elapsed = benchmark(
                routine,
                label,
                n,
                loops=nloops,
                generate_samples=generate_samples,
                **kwargs,
            )

            times.append(elapsed)
            if (nloops * elapsed > max_time) or (
                nloops * elapsed > red_time / 10 and len(Nloops) > 0
            ):
                nloops = Nloops.pop(0)

    except RuntimeError:
        print("**\nMemory overflow !")
        not_recorded_times = (len(problem_sizes) - len(times)) * [np.nan]

    except IndexError:  # Thrown by Nloops.pop(0) if Nloops = []
        print("**\nToo slow !")
        not_recorded_times = (len(problem_sizes) - len(times)) * [np.Infinity]

    return times + not_recorded_times


# Max number of seconds before we break the loop:
# Decrease the number of runs if computations take longer than 2s:


def full_benchmark(
    to_plot,
    routines,
    generate_samples,
    problem_sizes,
    min_time=1e-5,
    max_time=10,
    red_time=2,
    xlabel="Number of samples",
    ylabel="Time (s)",
    legend_location="upper left",
):

    print("Benchmarking : {} ===============================".format(to_plot))

    labels = [label for (_, label, _) in routines]

    lines = [problem_sizes] + [
        bench_config(
            *routine,
            generate_samples=generate_samples,
            problem_sizes=problem_sizes,
            max_time=max_time,
            red_time=red_time,
        )
        for routine in routines
    ]
    benches = np.array(lines).T

    # Creates a pyplot figure:
    plt.figure(figsize=(12, 8))
    linestyles = ["o-", "s-", "^-", "<-", ">-", "v-", "+-", "*-", "x-", "p-", "d-"]
    for i, label in enumerate(labels):
        plt.plot(
            benches[:, 0],
            benches[:, i + 1],
            linestyles[i % len(linestyles)],
            linewidth=2,
            label=label,
        )

        for (j, val) in enumerate(benches[:, i + 1]):
            if np.isnan(val) and j > 0:
                x, y = benches[j - 1, 0], benches[j - 1, i + 1]
                plt.annotate(
                    "Memory overflow!",
                    xy=(x, 1.05 * y),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )
                break

            elif np.isinf(val) and j > 0:
                x, y = benches[j - 1, 0], benches[j - 1, i + 1]
                plt.annotate(
                    "Too slow!",
                    xy=(x, 1.05 * y),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )
                break

    plt.title(to_plot)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(loc=legend_location)
    plt.grid(True, which="major", linestyle="-")
    plt.grid(True, which="minor", linestyle="dotted")
    plt.axis([problem_sizes[0], problem_sizes[-1], min_time, max_time])

    fmt = lambda x, pos: si_format(x, precision=0)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))

    fmt = lambda x, pos: si_format(x, precision=0) + "s"
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))

    plt.tight_layout()

    # Save as a .csv to put a nice Tikz figure in the papers:
    header = "Npoints, " + ", ".join(labels)
    os.makedirs("output", exist_ok=True)
    np.savetxt(
        "output/benchmark_convolutions_3D.csv",
        benches,
        fmt="%-9.5f",
        header=header,
        comments="",
        delimiter=",",
    )
