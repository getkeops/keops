import numpy as np

import pykeops.config
from pykeops.common.utils import pyKeOps_Message

formula = "SqNorm2(x - y)"
var = ["x = Vi(3)", "y = Vj(3)"]
expected_res = np.array([63.0, 90.0])


def test_numpy_bindings():
    """
    This function try to compile a simple keops formula using the numpy binder.
    """
    x = np.arange(1, 10).reshape(-1, 3).astype("float32")
    y = np.arange(3, 9).reshape(-1, 3).astype("float32")

    import pykeops.numpy as pknp

    my_conv = pknp.Genred(formula, var)
    if np.allclose(my_conv(x, y).flatten(), expected_res):
        pyKeOps_Message("pyKeOps with numpy bindings is working!", use_tag=False)
    else:
        pyKeOps_Message("outputs wrong values...", use_tag=False)


def test_torch_bindings():
    """
    This function try to compile a simple keops formula using the pytorch binder.
    """
    try:
        import torch
    except ImportError:
        pyKeOps_Message("torch not found...", use_tag=False)
        return
    except:
        pyKeOps_Message("unexpected error...", use_tag=False)
        return

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
