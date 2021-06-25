from keops.python_engine.formulas.VectorizedScalarOp import VectorizedScalarOp
from keops.python_engine.utils.math_functions import keops_cos


class Cos(VectorizedScalarOp):

    """the cosine vectorized operation"""

    string_id = "Cos"

    ScalarOpFun = keops_cos

    @staticmethod
    def Derivative(f):
        from .Sin import Sin

        return -Sin(f)
