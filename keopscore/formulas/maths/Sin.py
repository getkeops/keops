from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_sin


class Sin(VectorizedScalarOp):
    """the Sine vectorized operation"""

    string_id = "Sin"

    ScalarOpFun = keops_sin

    @staticmethod
    def Derivative(f):
        from .Cos import Cos

        return Cos(f)
