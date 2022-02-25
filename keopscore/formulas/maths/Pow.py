from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_pow


class Pow(VectorizedScalarOp):
    """the integer power vectorized operation
    Pow(f,m) where m is integer, computes f^m
    """

    def __init__(self, f, m):
        super().__init__(f, params=(m,))

    string_id = "Pow"

    ScalarOpFun = keops_pow

    @staticmethod
    def Derivative(f, m):
        from keopscore.formulas.variables.IntCst import IntCst

        return IntCst(m) * Pow(f, m - 1)

    # parameters for testing the operation (optional)
    nargs = 1  # number of arguments (excluding parameters)
    test_ranges = [(0, 2)]  # ranges of arguments
    test_params = [2]  # values of parameters for testing
    torch_op = "lambda x,m : torch.pow(x, m)"
