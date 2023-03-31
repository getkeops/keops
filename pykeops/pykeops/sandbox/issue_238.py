import pykeops
import torch
from pykeops.torch import Genred


def cauchy(v, z, w):
    expr = "ComplexDivide(v, z-w)"
    cauchy_mult = Genred(
        expr,
        [
            "v = Vj(2)",
            "z = Vi(2)",
            "w = Vj(2)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    v = torch.view_as_real(v)
    z = torch.view_as_real(z)
    w = torch.view_as_real(w)

    r = cauchy_mult(v, z, w, backend="GPU")
    return torch.view_as_complex(r)


def convert_data(*tensors, device="cuda"):
    """Prepare tensors for backwards pass"""
    tensors = tuple(t.to(device) for t in tensors)
    for t in tensors:
        if t.is_leaf:
            t.requires_grad = True
        t.retain_grad()
    return tensors


def data(B, N, L):
    w = torch.randn(B, N, dtype=torch.cfloat)
    v = torch.randn(B, N, dtype=torch.cfloat)
    z = torch.randn(B, L, dtype=torch.cfloat)

    w, v, z = convert_data(w, v, z)
    return w, v, z


def test():
    B = 4
    N = 64
    L = 256
    w, v, z = data(B, N, L)

    y = cauchy(v, z, w)
    print("output", y.shape, y.dtype)

    grad = torch.randn_like(y)
    y.backward(grad, retain_graph=True)

    print(grad.shape, grad.dtype)


if __name__ == "__main__":
    test()
