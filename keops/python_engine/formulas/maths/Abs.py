from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_abs
from keops.python_engine.formulas.maths.Sign import Sign


class Abs(VectorizedScalarOp):

    """the absolute value vectorized operation"""

    string_id = "Abs"

    ScalarOpFun = keops_abs

    @staticmethod
    def Derivative(f):
        return Sign(f)
