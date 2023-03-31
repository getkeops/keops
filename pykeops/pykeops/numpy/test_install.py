import numpy as np

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
