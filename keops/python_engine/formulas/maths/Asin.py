from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_asin
from keops.python_engine.formulas.maths.Rsqrt import Rsqrt


class Asin(VectorizedScalarOp):

    """the arc-sine vectorized operation"""

    string_id = "Asin"

    ScalarOpFun = keops_asin

    @staticmethod
    def Derivative(f):
        return Rsqrt(1 - f ** 2)
