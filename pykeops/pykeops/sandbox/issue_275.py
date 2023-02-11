import torch
from pykeops.torch import Genred


def _broadcast_dims(*args):
    out = []
    for x in args:
        out.append(x[:, None])
    return out


def _c2r(x):
    return torch.stack((torch.real(x), torch.imag(x)), dim=len(x.shape)).reshape(
        x.shape[:-1] + (-1,)
    )


def log_vandermonde(v, x, L, conj=True):
    expr = "ComplexMult(v, ComplexExp(ComplexMult(x, l)))"
    vandermonde_mult = Genred(
        expr,
        [
            "v = Vj(2)",
            "x = Vj(2)",
            "l = Vi(2)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    l = torch.arange(L).to(x)
    v, x, l = _broadcast_dims(v, x, l)
    v = _c2r(v)
    x = _c2r(x)
    l = _c2r(l)

    r = vandermonde_mult(v, x, l, backend="GPU")

    return r


L, N = 2000, 1000
v = torch.rand(N, dtype=torch.complex64)
x = 1j * torch.rand(N, dtype=torch.float32)

out = log_vandermonde(v, x, L)

print("ok")
print("shape of output : ", out.shape)
print("norm of output : ", torch.norm(out))
