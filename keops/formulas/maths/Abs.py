from keops.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.formulas.maths.Sign import Sign
from keops.utils.math_functions import keops_abs


class Abs(VectorizedScalarOp):
    """the absolute value vectorized operation"""

    string_id = "Abs"

    ScalarOpFun = keops_abs

    @staticmethod
    def Derivative(f):
        return Sign(f)
