from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_relu
from keops.python_engine.formulas.maths.Step import Step


class ReLU(VectorizedScalarOp):

    """the ReLU vectorized operation"""

    string_id = "ReLU"

    ScalarOpFun = keops_relu

    @staticmethod
    def Derivative(f):
        return Step(f)
