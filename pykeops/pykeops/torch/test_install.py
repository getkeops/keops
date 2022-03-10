import torch

from pykeops.common.utils import pyKeOps_Message

formula = "SqNorm2(x - y)"
var = ["x = Vi(3)", "y = Vj(3)"]
expected_res = [63.0, 90.0]


def test_torch_bindings():
    """
    This function try to compile a simple keops formula using the pytorch binder.
    """
    x = torch.arange(1, 10, dtype=torch.float32).view(-1, 3)
    y = torch.arange(3, 9, dtype=torch.float32).view(-1, 3)

    import pykeops.torch as pktorch

    my_conv = pktorch.Genred(formula, var)
    if torch.allclose(
        my_conv(x, y).view(-1), torch.tensor(expected_res).type(torch.float32)
    ):
        pyKeOps_Message("pyKeOps with torch bindings is working!", use_tag=False)
    else:
        pyKeOps_Message("outputs wrong values...", use_tag=False)
