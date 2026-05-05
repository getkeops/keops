import numpy as np

from pykeops.common.utils import pyKeOps_Message

formula = "SqNorm2(x - y)"
var = ["x = Vi(3)", "y = Vj(3)"]
expected_res = np.array([63.0, 90.0])


def test_numpy_bindings():
    """
    Try to compile a simple KeOps formula using the NumPy binder.
    """
    x = np.arange(1, 10).reshape(-1, 3).astype("float32")
    y = np.arange(3, 9).reshape(-1, 3).astype("float32")

    import pykeops.numpy as pknp

    my_conv = pknp.Genred(formula, var)

    try:
        keops_res = my_conv(x, y).flatten()
    except Exception as e:
        raise ValueError(f"Error during computation: {e}", use_tag=False)
    
    if np.allclose(keops_res, expected_res):
        pyKeOps_Message("pyKeOps with torch bindings is working!", use_tag=False, level=1)
        return True
    else:
        pyKeOps_Message(f"outputs wrong values: expected {expected_res} but get {keops_res}", use_tag=False, level=1)
        return False