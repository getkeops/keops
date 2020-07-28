"""
Comparison of conjugate gradient methods
==========================================

Different implementations of the conjugate gradient (CG) exist. Here, we compare the CG implemented in scipy which uses Fortran
against it's pythonized version and the older version of the algorithm available in pykeops.

We want to solve the positive definite linear system :math:`(K_{x,x} + \\alpha Id)a = b` for :math:`a, b, x \in \mathbb R^N`.

Let the Gaussian RBF kernel be defined as

.. math::

    K_{x,x}=\left[\exp(-\gamma \|x_i - x_j\|^2)\\right]_{i,j=1}^N. 


Choosing :math:`x` such that :math:`x_i = i/N,\ i=1,\dots, N` makes :math:`K_{x,x}` be a highly unwell-conditioned matrix for :math:`N\geq 10`.

"""

#############################
# Setup
# ----------
# Imports needed

import importlib
import os
import time
import inspect

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import aslinearoperator, cg

from pykeops.numpy import KernelSolve as KernelSolve_np, LazyTensor
from pykeops.torch import KernelSolve
from pykeops.torch.utils import squared_distances
from pykeops.torch import Genred as Genred_tch
from pykeops.numpy import Vi, Vj, Pm
from pykeops.numpy import Genred as Genred_np

use_cuda = torch.cuda.is_available()

device = torch.device("cuda") if use_cuda else torch.device("cpu")
print("The device used is {}.".format(device))

########################################
# Gaussian radial basis function kernel
########################################

formula = 'Exp(- g * SqDist(x,y)) * a' # linear w.r.t a
aliases = ['x = Vi(1)',   # First arg:  i-variable of size 1
           'y = Vj(1)',   # Second arg: j-variable of size 1
           'a = Vj(1)',  # Third arg:  j-variable of size 1
           'g = Pm(1)']


############################
# Functions to benchmark
###########################
#
# All systems are regularized with a ridge parameter ``alpha``. 
#
# The originals :
# 


def keops_tch(x, b, gamma, alpha):
    Kinv = KernelSolve(formula, aliases, "a", axis=1, dtype='float32')
    res = Kinv(x, x, b, gamma, alpha=alpha)
    return res


def keops_np(x, b, gamma, alpha, callback=None):
    Kinv = KernelSolve_np(formula, aliases, "a", axis=1, dtype='float32')
    res = Kinv(x, x, b, gamma, alpha=alpha, callback=callback)
    return res


####################################
# Scipy :
# 
#


def scipy_cg(x, b, gamma, alpha, callback=None):
    K_ij = (-Pm(gamma) * Vi(x).sqdist(Vj(x))).exp()
    A = aslinearoperator(
        diags(alpha * np.ones(x.shape[0]))) + aslinearoperator(K_ij)
    A.dtype = np.dtype('float32')
    res = cg(A, b, callback=callback)
    return res


####################################
# Pythonized scipy :
# 


def dic_cg_np(x, b, gamma, alpha, callback=None, check_cond=False):
    Kinv = KernelSolve_np(formula, aliases, "a", axis=1, dtype='float32')
    ans = Kinv.cg(x, x, b, gamma, alpha=alpha,
                      callback=callback, check_cond=check_cond)
    return ans


def dic_cg_tch(x, b, gamma, alpha, check_cond=False):
    Kinv = KernelSolve(formula, aliases, "a", axis=1, dtype='float32')
    ans = Kinv.cg(x, x, b, gamma, alpha=alpha, check_cond=check_cond)
    return ans


#########################
# Benchmarking
#########################

functions = [(scipy_cg, "numpy"),
             (keops_np, "numpy"), (keops_tch, "torch"),
             (dic_cg_np, "numpy"), (dic_cg_tch, "torch")]

sizes = [50,  100, 500, 1000, 5000, 20000, 40000]
reps = [50 ,  50 , 50,  10,   10,   5,     5]


def compute_error(func, pack, result, errors, x, b, alpha, gamma):
    if str(func)[10:15] == "keops":
        code = "a = func(x, b, gamma, alpha).reshape(b.shape);\
                err = ( (alpha * a + K(x, x, a, gamma) - b) ** 2).sum();\
                errors.append(err);"
    else:
        code = "a = func(x, b, gamma, alpha)[0].reshape(b.shape);\
                err = ( (alpha * a + K(x, x, a, gamma) - b) ** 2).sum();\
                errors.append(err);"

    if pack == 'numpy':
        K = Genred_np(formula, aliases, axis=1, dtype='float32')
    else:
        K = Genred_tch(formula, aliases, axis=1, dtype='float32')

    exec(code, locals())
    return errors


def to_bench(funcpack, size, rep):
    global use_cuda
    importlib.reload(torch)
    if device == 'cuda':
        torch.cuda.manual_seed_all(112358)
    else:
        torch.manual_seed(112358)
    code = "func(x, b, gamma, alpha)"
    func, pack = funcpack

    times = []
    errors = []

    if use_cuda:
        torch.cuda.synchronize()
    for i in range(rep):

        x = torch.linspace(1/size, 1, size, dtype=torch.float32,
                           device=device).reshape(size, 1)
        b = torch.randn(size, 1, device=device, dtype=torch.float32)
        # kernel bandwidth
        gamma = torch.ones(
            1, device=device, dtype=torch.float32) * .5 / .01 ** 2
        # regularization
        alpha = torch.ones(1, device=device, dtype=torch.float32) * 2

        if pack == 'numpy':
            x, b = x.cpu().numpy().astype("float32"), b.cpu().numpy().astype("float32")
            gamma, alpha = gamma.cpu().numpy().astype(
                "float32"), alpha.cpu().numpy().astype("float32")

        if i == 0:
            exec(code, locals())  # Warmup run, to compile and load everything

        start = time.perf_counter()
        result = func(x, b, gamma, alpha)
        if use_cuda:
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)
        errors = compute_error(func, pack, result, errors, x, b, alpha, gamma)

    return sum(times)/rep, sum(errors)/rep


def global_bench(functions, sizes, reps):
    list_times = [[] for _ in range(len(functions))]
    list_errors = [[] for _ in range(len(functions))]

    for j, one_to_bench in enumerate(functions):
        print("~~~~~~~~~~~~~Benchmarking {}~~~~~~~~~~~~~~.".format(one_to_bench))
        for i in range(len(sizes)):
            try:
                time, err = to_bench(one_to_bench, sizes[i], reps[i])
                list_times[j].append(time)
                list_errors[j].append(err)
            except:
                while len(list_times[j]) != len(reps):
                    list_times[j].append(np.nan)
                    list_errors[j].append(np.nan)
                break
            print("Finished size {}.".format(sizes[i]))

        print("Finished", one_to_bench[0], "in a cumulated time of {:3.9f}s.".format(
            sum(list_times[j])))

    return list_times, list_errors


#########################################
# Plot the results of the benchmarking
#########################################

list_times, list_errors = global_bench(functions, sizes, reps)
labels = ["scipy + keops", "keops_np", "keops_tch",
          "dico + keops_np", "dico + keops_tch"]

plt.style.use('ggplot')
plt.figure(figsize=(20,10))
plt.subplot(121)
for i in range(len(functions)):
    plt.plot(sizes, list_times[i], label=labels[i])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Kernel of size $n\times n$")
plt.ylabel("Computational time (s)")
plt.legend()
plt.subplot(122)
for i in range(len(functions)):
    plt.plot(sizes, list_errors[i], label=labels[i])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Kernel of size $n\times n$")
plt.ylabel(r"Error $||Ax_{k_{end}} -b||^2$")
plt.legend()
plt.tight_layout()
plt.show()


##############################################
# Stability
# ------------
#
# Stability of the errors and norm of the iterated approximations of the answer


def norm_stability(size, funcpack):
    errk_scipy, iter_scipy, x_scipy = [], [], []
    errk_dic, iter_dic, x_dic = [], [], []
    errk_keops, iter_keops, x_keops = [], [], []

    def callback_sci(xk):
        env = inspect.currentframe().f_back
        iter_scipy.append(env.f_locals['iter_'])
        x_scipy.append(env.f_locals['x'])
        err = ( ( alpha * xk.reshape(-1, 1) + K(x, x, xk.reshape(-1, 1), gamma) - b) ** 2).sum()
        errk_scipy.append(err)

    def callback_kinv_keops(xk):
        env = inspect.currentframe().f_back
        err = ( ( alpha * xk + K(x, x, xk, gamma) - b) ** 2).sum()
        errk_keops.append(err)
        iter_keops.append(env.f_locals['k'])
        x_keops.append(env.f_locals['a'])

    def callback_dic(xk):
        env = inspect.currentframe().f_back
        err = ( ( alpha * xk + K(x, x, xk, gamma) - b) ** 2).sum()
        errk_dic.append(err)
        iter_dic.append(env.f_locals['iter_'])
        x_dic.append(env.f_locals['x'])

    callback_list = [callback_sci, callback_kinv_keops, callback_dic]

    for i, funcpack in enumerate(funcpack):
        fun, pack = funcpack

        global x, b, gamma, alpha, K
        if device == 'cuda':
            torch.cuda.manual_seed_all(112358)
        else:
            torch.manual_seed(112358)

        x = torch.linspace(1/size, 1, size, dtype=torch.float32,
                           device=device).reshape(size, 1)
        b = torch.randn(size, 1, device=device, dtype=torch.float32)
        # kernel bandwidth
        gamma = torch.ones(
            1, device=device, dtype=torch.float32) * .5 / .01 ** 2
        # regularization
        alpha = torch.ones(1, device=device, dtype=torch.float32) * 2

        if pack == 'numpy':
            x, b = x.cpu().numpy().astype("float32"), b.cpu().numpy().astype("float32")
            gamma, alpha = gamma.cpu().numpy().astype(
                "float32"), alpha.cpu().numpy().astype("float32")
            K = Genred_np(formula, aliases, axis=1, dtype='float32')
        else:
            K = Genred_tch(formula, aliases, axis=1, dtype='float32')

        fun(x, b, gamma, alpha, callback=callback_list[i])
        del x, b, gamma, alpha, K
    return errk_scipy, iter_scipy, x_scipy, errk_dic, iter_dic, x_keops, errk_keops, iter_keops, x_dic


#########################################
# Plot the results of the stability
#########################################

onlynum = [(scipy_cg, "numpy"), (keops_np, "numpy"), (dic_cg_np, "numpy")]
errk_scipy, iter_scipy, x_scipy, errk_dic, iter_dic,\
    x_keops, errk_keops, iter_keops, x_dic = norm_stability(
        1000, onlynum)

scal_dic, scal_keops, scal_scipy = [], [], []
for i in range(1,len(iter_dic)):
    scal_dic.append((x_dic[i-1].T @ x_dic[i]).flatten())
for i in range(1, len(iter_keops)):
    scal_keops.append((x_keops[i-1].T @ x_keops[i]).flatten())
for i in range(1, len(iter_scipy)):
    scal_scipy.append((x_scipy[i-1].T @ x_scipy[i]).flatten())

plt.figure(figsize=(20,10))
plt.subplot(121)
plt.plot(iter_keops, errk_keops, 'o-', label=labels[1])
plt.plot(iter_scipy, errk_scipy, '^-', label=labels[0])
plt.plot(iter_dic, errk_dic, 'x-', label=labels[3])
plt.yscale('log')
plt.xlabel(r"Iteration k")
plt.ylabel(r"$||(\alpha\ Id + K_{x,x})x_k - b||^2$")
plt.legend()

plt.subplot(122)
plt.plot(iter_keops[1:], scal_keops, 'o-', label=labels[1])
plt.plot(iter_scipy[1:], scal_scipy, '^-', label=labels[0])
plt.plot(iter_dic[1:], scal_dic, 'x-', label=labels[3])
plt.yscale('log')
plt.xlabel(r"Iteration k")
plt.ylabel(r"$\langle x_{k-1}|x_k\rangle $")
plt.legend()

plt.tight_layout()
plt.show()

####################################################### 
# Condition number check
# -------------------------------------
#
# Scipy's algorithm can't be used practically for large kernels in this case. The condition number can be why.
#
#
# The argument ``check_cond`` in Keops lets the user have an idea of the conditioning number of the matrix :math:`A=(K_{x,x} + \alpha Id)`. A warning appears
# if :math:`\mathrm{cond}(A)>500`. The user is also warned if the CG algorithm reached its maximum number of iterations *ie* did not converge. The idea here
# is not to estimate the condition number and let the user have another sanity check at disposal.
#
# To test the condition number :math:`\mathrm{cond}(A)=\frac{\lambda_{\max}}{\lambda_{\min}}`, we first use the
# power iteration to have a good estimation of :math:`\lambda_{\max}`. Then, wee apply the inverse power iteration
# to obtain the iterations :math:`\mu_k` of the estimated :math:`\lambda_{\min}` using the Rayleigh's quotient after having the iterations :math:`u_k`
# of the estimated eigen vector :math:`u_1`. The distance between the vectors :math:`v_k` and :math:`u_1` decreasing over the iterations at a rate of
# :math:`\mathcal{O}\left(\left|\frac{\lambda_{\min}}{\lambda_{submin}}\right|^k\right)`, if we don't want
# :math:`\frac{\lambda_{\max}}{\lambda_{\min}}>500` then :math:`\mu_k` must not be below the threshold :math:`\frac{\lambda_{\max}}{500}`
# If so, the system warns the user that the condition number might be too high.
#
# In practice only a few iterations are necessary to go below this threshold. Thus we fixed a maximum number of iterations for the inverse
# power method to ``50`` so that for large matrices it doesn't take too much time.


def test_cond(device, size, pack, alpha):
    if device == 'cuda':
        torch.cuda.manual_seed_all(1234)
    else:
        torch.manual_seed(1234)

    x = torch.linspace(1/size, 1, size, dtype=torch.float32,
                       device=device).reshape(size, 1)
    b = torch.randn(size, 1, device=device, dtype=torch.float32)
    # kernel bandwidth
    gamma = torch.ones(1, device=device, dtype=torch.float32) * .5 / .01 ** 2
    alpha = torch.ones(1, device=device, dtype=torch.float32) * alpha  # regularization

    if pack == 'numpy':
        x, b = x.cpu().numpy().astype("float32"), b.cpu().numpy().astype("float32")
        gamma, alpha = gamma.cpu().numpy().astype(
            "float32"), alpha.cpu().numpy().astype("float32")
        ans = dic_cg_np(x, b, gamma, alpha, check_cond=True)
    else:
        ans = dic_cg_tch(x, b, gamma, alpha, check_cond=True)
    return ans


print("Condition number warnings tests")
print("Small matrix well conditioned (nothing should appear)")
ans = test_cond(device, 20, 'numpy', alpha=1)
print("Large matrix unwell conditioned (a warning should appear)")
ans2 = test_cond(device, 1000, 'numpy', alpha=1e-6)
print("Large matrix unwell conditioned but with a large regularization (nothing should appear)")
ans3 = test_cond(device, 1000, 'numpy', alpha=100)


##########################
# Zoom in on Keops times
############################
#
# Let's consider the Keops conjugate gradients for large kernels. Scipy's algorithm explodes in time for
# :math:`n\geq 50000` so we only consider the keops implementations here.
#


functions = functions[1:]
sizes = [30000, 50000, 100000, 200000]
reps = [5,     5,     5,      2]
list_times, list_errors = global_bench(functions, sizes, reps)
labels = ["keops_np", "keops_tch",
          "dico + keops_np", "dico + keops_tch"]
plt.style.use('ggplot')
plt.figure(figsize=(20,10))
plt.subplot(121)
for i in range(len(functions)):
    plt.plot(sizes, list_times[i], label=labels[i])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Kernel of size $n\times n$")
plt.ylabel("Computational time (s)")
plt.legend()
plt.subplot(122)
for i in range(len(functions)):
    plt.plot(sizes, list_errors[i], label=labels[i])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Kernel of size $n\times n$")
plt.ylabel(r"Error $||Ax_{k_{end}} -b||^2$")
plt.legend()
plt.show()

###########################
# Random points
###########################
#
# Let's now use random values for :math:`x_i`.

def to_bench(funcpack, size, rep):
    global use_cuda
    importlib.reload(torch)
    if device == 'cuda':
        torch.cuda.manual_seed_all(112358)
    else:
        torch.manual_seed(112358)
    code = "func(x, b, gamma, alpha)"
    func, pack = funcpack

    times = []
    errors = []

    if use_cuda:
        torch.cuda.synchronize()
    for i in range(rep):

        x = torch.linspace(1/size, 1, size, dtype=torch.float32,
                           device=device).reshape(size, 1)
        b = torch.randn(size, 1, device=device, dtype=torch.float32)
        # kernel bandwidth
        gamma = torch.ones(
            1, device=device, dtype=torch.float32) * .5 / .01 ** 2
        # regularization
        alpha = torch.ones(1, device=device, dtype=torch.float32) * 2

        if pack == 'numpy':
            x, b = x.cpu().numpy().astype("float32"), b.cpu().numpy().astype("float32")
            gamma, alpha = gamma.cpu().numpy().astype(
                "float32"), alpha.cpu().numpy().astype("float32")

        if i == 0:
            exec(code, locals())  # Warmup run, to compile and load everything

        start = time.perf_counter()
        result = func(x, b, gamma, alpha)
        if use_cuda:
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)
        errors = compute_error(func, pack, result, errors, x, b, alpha, gamma)

    return sum(times)/rep, sum(errors)/rep


functions = [(scipy_cg, "numpy"),
             (keops_np, "numpy"), (keops_tch, "torch"),
             (dic_cg_np, "numpy"), (dic_cg_tch, "torch")]


sizes = [50,  100, 500, 1000, 5000, 20000, 40000]
reps = [50 ,  50 , 50,  10,   10,   5,     5]

list_times, list_errors = global_bench(functions, sizes, reps)
labels = ["scipy + keops", "keops_np", "keops_tch",
          "dico + keops_np", "dico + keops_tch"]

plt.style.use('ggplot')
plt.figure(figsize=(20,10))
plt.subplot(121)
for i in range(len(functions)):
    plt.plot(sizes, list_times[i], label=labels[i])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Kernel of size $n\times n$")
plt.ylabel("Computational time (s)")
plt.legend()
plt.subplot(122)
for i in range(len(functions)):
    plt.plot(sizes, list_errors[i], label=labels[i])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Kernel of size $n\times n$")
plt.ylabel(r"Error $||Ax_{k_{end}} -b||^2$")
plt.legend()
plt.tight_layout()
plt.show()