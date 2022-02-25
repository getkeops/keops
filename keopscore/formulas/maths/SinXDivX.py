from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.maths.Cos import Cos
from keopscore.formulas.maths.Sin import Sin
from keopscore.utils.math_functions import keops_sinxdivx


class SinXDivX(VectorizedScalarOp):
    """
    the sin(x)/x vectorized operation
    """

    string_id = "SinXDivX"

    ScalarOpFun = keops_sinxdivx

    @staticmethod
    def Derivative(f):
        return Cos(f) / f - Sin(f) / f**2
