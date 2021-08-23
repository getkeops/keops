from keops.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.formulas.maths.Step import Step
from keops.utils.math_functions import keops_relu


class ReLU(VectorizedScalarOp):
    """the ReLU vectorized operation"""

    string_id = "ReLU"

    ScalarOpFun = keops_relu

    @staticmethod
    def Derivative(f):
        return Step(f)
