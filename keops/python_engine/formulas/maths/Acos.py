from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_acos
from keops.python_engine.formulas.maths.Rsqrt import Rsqrt


class Acos(VectorizedScalarOp):

    """the arc-cosine vectorized operation"""

    string_id = "Acos"

    ScalarOpFun = keops_acos

    @staticmethod
    def Derivative(f):
        return -Rsqrt(1 - f ** 2)
