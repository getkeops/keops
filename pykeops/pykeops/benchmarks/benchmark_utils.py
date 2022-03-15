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

# import jax

use_cuda = torch.cuda.is_available()

##################################################
# Utility functions:
#


def timer(use_torch=True):
    if use_cuda and use_torch:
        torch.cuda.synchronize()
    return time.perf_counter()


def flatten(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]


def clear_gpu_cache():
    if use_cuda:
        torch.cuda.empty_cache()


################################################
# Timeout helper:
#

from functools import wraps
import errno
import signal


class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


##################################################
# Conversion routines:
#


def tensor(*x):
    if use_cuda:
        return torch.cuda.FloatTensor(*x)
    else:
        return torch.FloatTensor(*x)


def int_tensor(*x):
    if use_cuda:
        return torch.cuda.LongTensor(*x)
    else:
        return torch.LongTensor(*x)


def jax_tensor(*x):
    import jax

    return jax.device_put(*x)


#####################################################
# Random samples:
#


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


###########################################
# Multiprocessing code
# -----------------------------------------
#
#
# Unfortunately, some FAISS routines throw a C++ "abort" signal instead
# of a proper Python exception for out of memory errors on large problems.
# Letting them run in a separate process is the only way of handling
# the error without aborting the full benchmark.

import multiprocess as mp
import traceback
import queue
import sys
import uuid


def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)

    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


class Process(mp.Process):
    """Exception-friendly process class."""

    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def with_queue(f, queue, points):
    o = f(points)
    queue.put(o)


def run_safely(f, x):
    """Runs f(args) in a separate process."""

    # f_global = f # globalize(f)
    # mp.freeze_support()
    mp.set_start_method("spawn")
    q = mp.Queue()
    p = Process(target=with_queue, args=(f, q, x))
    p.start()
    p.join()

    if p.exception:
        error, traceback = p.exception
        print(traceback)
        raise error

    try:
        out = q.get(False, 2.0)  # Non-blocking mode
    except queue.Empty:
        print("Empty queue!")
        print("Exit code: ", p.exitcode)
        raise MemoryError()

    return out


##############################################
# Benchmarking loops
# -----------------------


def simple_loop(N, loops, routine, max_time, args, kwargs):
    # Warmup run, to compile and load everything:
    output = routine(*args, **kwargs)

    t_0 = timer()
    for i in range(loops):
        output = routine(*args, **kwargs)
    elapsed = timer() - t_0

    B = kwargs.get("batchsize", 1)
    perf = elapsed / (B * loops)
    print(
        f"{B:3}x{loops:3} loops of size {si_format(N,precision=0):>5}: {B:3}x{loops:3}x {si_format(perf):>7}s"
    )

    return perf


def recall(out_indices, true_indices):
    Ntest, K = out_indices.shape
    true_indices = true_indices[:Ntest, :K]
    r = 0.0
    for k in range(Ntest):
        r += np.sum(np.in1d(out_indices[k], true_indices[k], assume_unique=True)) / K
    r /= Ntest
    return r


def train_test_loop(N, loops, routine, max_time, args, kwargs):

    x_train = args["train"]
    x_test = args["test"]
    ground_truth = args["output"]

    # Warmup run, to compile and load everything:
    operator = routine(N, **args, **kwargs)
    clear_gpu_cache()
    model, _ = timeout(6 * max_time)(operator)(x_train)
    clear_gpu_cache()
    output, _ = timeout(max_time)(model)(x_test)

    # Time the training step:
    train_time = 0.0
    for i in range(loops):
        clear_gpu_cache()
        model, elapsed = operator(x_train)
        train_time += elapsed

    # Time the test step:
    test_time = 0.0
    for i in range(loops):
        clear_gpu_cache()
        output, elapsed = model(x_test)
        test_time += elapsed

    B = kwargs.get("batchsize", 1)
    train_perf = train_time / (B * loops)
    test_perf = test_time / (B * loops)
    perf = recall(output, ground_truth)

    print(f"{B:3}x{loops:3} loops of size {si_format(N,precision=0):>5}: ", end="")
    print(f"train = {B:3}x{loops:3}x {si_format(train_perf):>7}s, ", end="")
    print(f"test  = {B:3}x{loops:3}x {si_format(test_perf):>7}s, ", end="")
    print(f"recall = {100*perf:>3.0f}%")

    if perf < 0.75:
        raise ValueError("** Recall lower than 75%!")

    return test_perf


def benchmark(
    routine,
    label,
    N,
    max_time,
    loops=10,
    generate_samples=None,
    **kwargs,
):

    importlib.reload(torch)  # In case we had a memory overflow just before...
    args = generate_samples(N, **kwargs)

    benchmark_loop = train_test_loop if type(args) is dict else simple_loop

    # Actual benchmark:
    elapsed = benchmark_loop(N, loops, routine, max_time, args, kwargs)
    return elapsed


def bench_config(
    routine,
    label,
    kwargs,
    generate_samples=None,
    problem_sizes=[1],
    max_time=10,
    red_time=2,
    loops=[100, 10, 1],
):
    """Times a convolution for an increasing number of samples."""

    print(f"{label} -------------")

    times = []
    not_recorded_times = []
    try:
        Nloops = loops.copy()
        nloops = Nloops.pop(0)
        for n in problem_sizes:

            elapsed = benchmark(
                routine,
                label,
                n,
                max_time,
                loops=nloops,
                generate_samples=generate_samples,
                **kwargs,
            )

            times.append(elapsed)
            if (nloops * elapsed > max_time) or (
                nloops * elapsed > red_time and len(Nloops) > 0
            ):
                nloops = Nloops.pop(0)

    except MemoryError:
        print("** Memory overflow!")
        not_recorded_times = (len(problem_sizes) - len(times)) * [np.nan]

    except (TimeoutError, IndexError):  # Thrown by Nloops.pop(0) if Nloops = []
        print("** Too slow!")
        not_recorded_times = (len(problem_sizes) - len(times)) * [np.Infinity]

    except NotImplementedError:
        print("** This metric is not supported!")
        not_recorded_times = (len(problem_sizes) - len(times)) * [np.Infinity]

    except ValueError as err:
        print(err)
        not_recorded_times = (len(problem_sizes) - len(times)) * [np.NINF]

    except RuntimeError as err:
        print(err)
        print("** Runtime error!")
        not_recorded_times = (len(problem_sizes) - len(times)) * [np.nan]

    return times + not_recorded_times


def identity(x):
    return x


def queries_per_second(N):
    def qps(x):
        return N / x

    return qps


def inf_to_nan(x):
    y = x.copy()
    y[~np.isfinite(y)] = np.nan
    return y


def full_benchmark(
    to_plot,
    routines,
    generate_samples,
    problem_sizes,
    min_time=1e-5,
    max_time=10,
    red_time=2,
    loops=[100, 10, 1],
    xlabel="Number of samples",
    ylabel="Time (s)",
    frequency=False,
    legend_location="upper left",
    linestyles=["o-", "s-", "^-", "<-", ">-", "v-", "+-", "*-", "x-", "p-", "d-"],
):

    if frequency:
        N = len(generate_samples(1)["test"])
        transform = queries_per_second(N)
        ymin, ymax = transform(max_time), transform(min_time)
        y_suffix = "Hz"
    else:
        transform = identity
        ymin, ymax = min_time, max_time
        y_suffix = "s"

    print("Benchmarking : {} ===============================".format(to_plot))

    labels = [label for (_, label, _) in routines]

    lines = [problem_sizes] + [
        bench_config(
            *routine,
            generate_samples=generate_samples,
            problem_sizes=problem_sizes,
            max_time=max_time,
            red_time=red_time,
            loops=loops,
        )
        for routine in routines
    ]
    benches = np.array(lines).T

    # Creates a pyplot figure:
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(labels):
        plt.plot(
            benches[:, 0],
            transform(inf_to_nan(benches[:, i + 1])),
            linestyles[i % len(linestyles)],
            linewidth=2,
            label=label,
        )

        for (j, val) in enumerate(benches[:, i + 1]):
            if np.isnan(val) and j > 0:
                x, y = benches[j - 1, 0], transform(benches[j - 1, i + 1])
                plt.annotate(
                    "Memory overflow!",
                    xy=(1.05 * x, y),
                    horizontalalignment="left",
                    verticalalignment="center",
                )
                break

            elif np.isposinf(val) and j > 0:
                x, y = benches[j - 1, 0], transform(benches[j - 1, i + 1])
                plt.annotate(
                    "Too slow!",
                    xy=(1.05 * x, y),
                    horizontalalignment="left",
                    verticalalignment="center",
                )
                break

            elif np.isneginf(val) and j > 0:
                x, y = benches[j - 1, 0], transform(benches[j - 1, i + 1])
                plt.annotate(
                    "Recall < 75%",
                    xy=(1.05 * x, y),
                    horizontalalignment="left",
                    verticalalignment="center",
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
    plt.axis([problem_sizes[0], problem_sizes[-1], ymin, ymax])

    fmt = lambda x, pos: si_format(x, precision=0)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))

    fmt = lambda x, pos: si_format(x, precision=0) + y_suffix
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))

    # plt.tight_layout()

    # Save as a .csv to put a nice Tikz figure in the papers:
    header = "Npoints, " + ", ".join(labels)
    os.makedirs("output", exist_ok=True)
    np.savetxt(
        f"output/{to_plot}.csv",
        benches,
        fmt="%-9.5f",
        header=header,
        comments="",
        delimiter=",",
    )
