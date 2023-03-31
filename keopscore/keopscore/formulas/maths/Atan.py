from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_atan


class Atan(VectorizedScalarOp):
    """the arc-tangent vectorized operation"""

    string_id = "Atan"

    ScalarOpFun = keops_atan

    @staticmethod
    def Derivative(f):
        return 1 / (1 + f**2)
