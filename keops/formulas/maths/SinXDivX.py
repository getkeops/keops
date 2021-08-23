from keops.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.formulas.maths.Cos import Cos
from keops.formulas.maths.Sin import Sin
from keops.utils.math_functions import keops_sinxdivx


class SinXDivX(VectorizedScalarOp):
    """the sin(x)/x vectorized operation"""

    string_id = "SinXDivX"

    ScalarOpFun = keops_sinxdivx

    @staticmethod
    def Derivative(f):
        return Cos(f) / f - Sin(f) / f ** 2
