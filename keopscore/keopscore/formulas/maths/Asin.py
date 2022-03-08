from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.maths.Rsqrt import Rsqrt
from keopscore.utils.math_functions import keops_asin


class Asin(VectorizedScalarOp):
    """the arc-sine vectorized operation"""

    string_id = "Asin"

    ScalarOpFun = keops_asin

    @staticmethod
    def Derivative(f):
        return Rsqrt(1 - f**2)

    # parameters for testing the operation (optional)
    test_ranges = [(-1, 1)]  # range of argument
