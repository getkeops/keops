import torch

from pykeops.common.utils import pyKeOps_Message

formula = "SqNorm2(x - y)"
var = ["x = Vi(3)", "y = Vj(3)"]
expected_res = [63.0, 90.0]


def test_torch_bindings():
    """
    Try to compile a simple KeOps formula using the PyTorch binder.
    """
    x = torch.arange(1, 10, dtype=torch.float32).view(-1, 3)
    y = torch.arange(3, 9, dtype=torch.float32).view(-1, 3)

    import pykeops.torch as pktorch

    my_conv = pktorch.Genred(formula, var)

    try:
        keops_res = my_conv(x, y).view(-1)
    except Exception as e:
        pyKeOps_Message(f"Error during computation: {e}", use_tag=False)
        return False

    if torch.allclose(keops_res, torch.tensor(expected_res, dtype=torch.float32)):
        pyKeOps_Message("pyKeOps with torch bindings is working!", use_tag=False)
        return True
    else:
        pyKeOps_Message(
            f"outputs wrong values: expected {expected_res} but get {keops_res}",
            use_tag=False,
        )
        return False
