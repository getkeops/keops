import torch
import pykeops
from pykeops.torch import Genred


def kernel(v):
    formula = "Rsqrt(v)"
    fn = Genred(formula, ["v=Vi(2)"], reduction_op="Sum", axis=1)
    res = fn(v, backend="CPU")
    return res


def test():
    assert torch.cuda.is_available
    v = torch.randn(100, 10)
    return kernel(v)


if __name__ == "__main__":
    test()
