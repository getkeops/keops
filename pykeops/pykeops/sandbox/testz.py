import torch


def fun1(x):
    return (x**2).sum()


def fun2(x):
    K = (x**2).sum(-1)
    K = K[..., None]
    res = (K.sum(2)).sum(3)
    return res


x = torch.rand(4, 4, 10, 3, 2)
hess1 = torch.func.hessian(fun1)(x)
hess2 = torch.func.hessian(fun2)(x)
print(hess2.shape)

print("fun1:", hess1.shape)
print("fun2:", hess2.shape)
