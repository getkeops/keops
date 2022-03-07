from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_powf


class Powf(VectorizedScalarOp):
    """the Power vectorized operation"""

    string_id = "Powf"

    ScalarOpFun = keops_powf

    @staticmethod
    def Derivative(a, b):
        from keopscore.formulas.maths.Log import Log

        return b * Powf(a, b - 1), Log(a) * Powf(a, b)

    # parameters for testing the operation (optional)
    test_ranges = [(0, 2), (-1, 1)]  # ranges of arguments
    torch_op = "lambda x,y : torch.pow(x, y)"
