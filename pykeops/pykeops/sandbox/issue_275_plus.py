import torch
from pykeops.torch import Genred
from time import time


def _broadcast_dims(*args):
    out = []
    for x in args:
        out.append(x[:, None])
    return out


def _c2r(x):
    return torch.stack((torch.real(x), torch.imag(x)), dim=len(x.shape)).reshape(
        x.shape[:-1] + (-1,)
    )


def log_vandermonde(v, x, L):
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


vandermonde_mult_alt = Genred(
    "ComplexMult(v, ComplexExp1j(Mult(x, l)))",
    [
        "v = Vj(2)",
        "x = Vj(1)",
        "l = Vi(1)",
    ],
    reduction_op="Sum",
    axis=1,
)


def log_vandermonde_alt(v, x, L):
    l = torch.arange(L).to(x)
    v, x, l = _broadcast_dims(v, x, l)
    v = _c2r(v)

    r = vandermonde_mult_alt(v, x, l, backend="GPU")

    return r


L, N = 20000, 10000
v = torch.rand(N, dtype=torch.complex64)
x = torch.rand(N, dtype=torch.float32)
xc = 1j * x

out = log_vandermonde(v, xc, L)
out_alt = log_vandermonde_alt(v, x, L)

start = time()
out = log_vandermonde(v, xc, L)
end = time()
print("time for original : ", end - start)
start = time()
out_alt = log_vandermonde_alt(v, x, L)
end = time()
print("time for alt : ", end - start)

print("norm of diff : ", torch.norm(out - out_alt))
