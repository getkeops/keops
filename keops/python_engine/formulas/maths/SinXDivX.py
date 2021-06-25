from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_sinxdivx
from keops.python_engine.formulas.maths.Cos import Cos
from keops.python_engine.formulas.maths.Sin import Sin


class SinXDivX(VectorizedScalarOp):

    """the sin(x)/x vectorized operation"""

    string_id = "SinXDivX"

    ScalarOpFun = keops_sinxdivx

    @staticmethod
    def Derivative(f):
        return Cos(f) / f - Sin(f) / f ** 2
