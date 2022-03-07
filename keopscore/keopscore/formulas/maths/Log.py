from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_log


class Log(VectorizedScalarOp):
    """the logarithm vectorized operation"""

    string_id = "Log"

    ScalarOpFun = keops_log

    @staticmethod
    def Derivative(f):
        return 1 / f

    # parameters for testing the operation (optional)
    test_ranges = [(0, 2)]  # range of argument
