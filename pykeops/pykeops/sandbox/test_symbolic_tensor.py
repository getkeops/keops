import torch
from pykeops.symbolictensor.pytorch import keops

# test for symbolic tensors


def TestFun(fun, *args):
    # compare KeOps and PyTorch implementation
    def compare_outputs(x, y):
        if x.shape == y.shape:
            print("  output shapes are the same.")
            rel_err = torch.norm(x - y) / torch.norm(x)
            print(f"  relative error is {rel_err}")
        else:
            print("  output shapes are different.")

    print(f"Testing function {fun.__name__}:")
    res_torch = fun(*args)
    res_keops = keops(fun)(*args)
    compare_outputs(res_torch, res_keops)
    gradin = torch.rand(res_torch.shape)
    args_rg = [arg for arg in args if arg.requires_grad]
    grads_torch = torch.autograd.grad(res_torch, args_rg, gradin)
    grads_keops = torch.autograd.grad(res_keops, args_rg, gradin)
    print("Testing gradients:")
    for k, arg in enumerate(args_rg):
        compare_outputs(grads_torch[k], grads_keops[k])


def fun1(x, y, b):
    K = (x - y) ** 2 / b
    return K.sum(axis=1)


def fun2(x, y, b):
    K = (x - y) ** 2 - b
    K = K.sum(axis=3)
    return K.sum(axis=1)


import torch


def varset1():
    M, N, D1, D2 = 5, 4, 3, 2
    x = torch.rand(M, 1, D1, D2, requires_grad=False)
    y = torch.rand(1, N, 1, D2)
    b = torch.rand(1, 1, 1, D2)
    return x, y, b


def varset2():
    M, N, D = 5, 4, 3
    x = torch.rand(M, 1, D, requires_grad=True)
    y = torch.rand(1, N, 1, requires_grad=True)
    b = torch.rand(1, 1, D, requires_grad=True)
    return x, y, b


x, y, b = varset2()

TestFun(fun1, x, y, b)
