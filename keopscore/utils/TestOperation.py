import time
import torch
import numpy as np
from torch.autograd import grad
from pykeops.torch import Genred
from keopscore.formulas import *
import types


def TestOperation(op_str, tol=1e-4, dtype="float32", test_grad=True):

    # N.B. dtype can be 'float32', 'float64' or 'float16'

    print("")

    keops_op = eval(op_str)
    if isinstance(keops_op, types.FunctionType):
        op_class_str = f"{op_str}_Impl"
        exec(f"from {keops_op.__module__} import {op_class_str}")
        keops_op_class = eval(op_class_str)
    else:
        keops_op_class = keops_op

    if keops_op_class.enable_test:
        if hasattr(keops_op_class, "test_argdims"):
            dims = keops_op_class.test_argdims
            nargs = len(dims)
        else:
            if hasattr(keops_op_class, "nargs"):
                nargs = keops_op_class.nargs
            elif hasattr(keops_op_class, "test_ranges"):
                nargs = len(keops_op_class.test_ranges)
            elif hasattr(keops_op_class, "Derivative"):
                from inspect import signature

                nargs = len(signature(keops_op_class.Derivative).parameters)
                if hasattr(keops_op_class, "test_params"):
                    nargs -= len(keops_op_class.test_params)
            else:
                print("no test available for " + op_str)
                return None
            dims = [3] * nargs
    else:
        print("no test available for " + op_str)
        return None

    #####################################################################
    # Declare random inputs:

    M = 3000
    N = 5000

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

    argcats = np.random.choice(["i", "j"], nargs)

    if hasattr(keops_op_class, "test_ranges"):
        rng = keops_op_class.test_ranges
        rand = torch.rand
    else:
        rng = [(0, 1)] * nargs
        rand = torch.randn

    args = [None] * nargs
    for k in range(nargs):
        MorN = M if argcats[k] == "i" else N
        args[k] = rand(
            MorN, dims[k], dtype=torchtype, device=device, requires_grad=True
        )
        args[k] = args[k] * (rng[k][1] - rng[k][0]) + rng[k][0]

    ####################################################################
    # Define a custom formula
    # -----------------------

    if hasattr(keops_op_class, "test_params"):
        params = keops_op_class.test_params
    else:
        params = ()

    formula = (
        op_str
        + "("
        + ",".join(f"v{k}" for k in range(nargs))
        + ","
        + ",".join(str(p) for p in params)
        + ")"
    )

    variables = list(f"v{k} = V{argcats[k]}({dims[k]})" for k in range(nargs))

    print("Testing operation " + op_str)

    my_routine = Genred(formula, variables, reduction_op="Sum", axis=1)
    c = my_routine(*args)

    print("ok, no error")
    print("5 first values :", *c.flatten()[:5].tolist())

    ####################################################################
    # Compute the gradient
    # -----------------------

    if test_grad:
        e = torch.rand_like(c)

        print("Testing gradient of operation " + op_str)

        g = grad(c, args, e)

        print("ok, no error")
        for k in range(nargs):
            app_str = f"number {k}" if len(args) > 1 else ""
            print(
                f"5 first values for gradient {app_str}:", *g[k].flatten()[:5].tolist()
            )

    if not hasattr(keops_op_class, "torch_op"):
        torch_op_str = keops_op_class.string_id.lower()
        if not torch_op_str in dir(torch):
            return "no_torch"
        torch_op = "torch." + torch_op_str
    else:
        if keops_op_class.torch_op is None:
            return "no_torch"
        torch_op = keops_op_class.torch_op

    print("Comparing with PyTorch implementation ")

    torch_args = [None] * nargs
    for k in range(nargs):
        torch_args[k] = (
            args[k][:, None, :] if argcats[k] == "i" else args[k][None, :, :]
        )

    ####################################################################
    # The equivalent code with a "vanilla" pytorch implementation

    if isinstance(torch_op, str):
        torch_op = eval(torch_op)

    c_torch = torch_op(*torch_args, *params).sum(dim=1)
    err_op = torch.norm(c - c_torch).item() / torch.norm(c_torch).item()
    print("relative error for operation :", err_op)

    if test_grad:
        if (
            not hasattr(keops_op_class, "no_torch_grad")
            or not keops_op_class.no_torch_grad
        ):
            g_torch = grad(c_torch, args, e)
            err_gr = [None] * nargs
            for k in range(nargs):
                app_str = f"number {k}" if len(args) > 1 else ""
                err_gr[k] = (
                    torch.norm(g[k] - g_torch[k]) / torch.norm(g_torch[k])
                ).item()
                print(f"relative error for gradient {app_str}:", err_gr[k])
        else:
            print("no gradient for torch")
            return abs(err_op) > tol
        return abs(err_op) > tol, list(abs(err) > tol for err in err_gr)
    else:
        return abs(err_op) > tol
