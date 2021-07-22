from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_powf


class Powf(VectorizedScalarOp):
    """the Power vectorized operation"""

    string_id = "Powf"

    ScalarOpFun = keops_powf

    @staticmethod
    def Derivative(a, b):
        from keops.python_engine.formulas.maths.Log import Log

        return b * Powf(a, b - 1), Log(a) * Powf(a, b)
