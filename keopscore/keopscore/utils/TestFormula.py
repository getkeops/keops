from time import time
import torch
import numpy as np
from torch.autograd import grad
from pykeops.torch import Genred
from keopscore.formulas import *


def TestFormula(formula, dtype="float32", test_grad=False, randseed=None):
    # N.B. dtype can be 'float32', 'float64' or 'float16'

    if isinstance(formula, str):
        formula_str = formula
        formula = eval(formula_str)
    else:
        formula_str = formula.__repr__()

    if randseed is not None:
        torch.manual_seed(randseed)

    print("")

    formula = eval(formula_str)
    vars = formula.Vars_
    nargs = 1 + max(var.ind for var in vars)

    #####################################################################
    # Declare random inputs:

    M = 300000
    N = 500000

    # Choose the storage place for our data : CPU (host) or GPU (device) memory.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype == "float32":
        torchtype = torch.float32
    elif dtype == "float64":
        torchtype = torch.float64
    elif dtype == "float16":
        torchtype = torch.float16
    else:
        raise ValueError("invalid dtype")

    rand = torch.randn

    args = [None] * nargs
    for var in vars:
        MorN = M if var.cat == 0 else N
        args[var.ind] = rand(
            MorN, var.dim, dtype=torchtype, device=device, requires_grad=True
        )

    print("Testing formula " + formula_str)

    my_routine = Genred(formula_str, [], reduction_op="Sum", axis=1)
    c = my_routine(*args)

    print("ok, no error")
    start = time()
    for k in range(10):
        c = my_routine(*args)
    end = time()
    print("average time over 10 calls:", (end-start)/10)
    print("5 first values :", *c.flatten()[:5].tolist())

    ####################################################################
    # Compute the gradient
    # -----------------------

    if test_grad:
        e = torch.rand_like(c)

        print("Testing gradient of formula " + formula_str)

        g = grad(c, args, e)

        print("ok, no error")
        start = time()
        for k in range(10):
            g = grad(c, args, e)
        end = time()
        print("average time over 10 calls:", (end-start)/10)

        for k in range(nargs):
            app_str = f"number {k}" if len(args) > 1 else ""
            print(
                f"5 first values for gradient {app_str}:", *g[k].flatten()[:5].tolist()
            )
        return c, g
    return c
