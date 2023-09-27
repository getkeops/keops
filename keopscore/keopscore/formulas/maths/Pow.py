from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_pow


class Pow_Impl(VectorizedScalarOp):
    """the integer power vectorized operation
    Pow(f,m) where m is integer, computes f^m
    """

    def __init__(self, f, m=None, params=None):
        # N.B. init via params keyword is used for compatibility with base class.
        if m is None:
            # here params should be a tuple containing one single integer
            (m,) = params
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


# N.B. The following separate function could theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Pow(f, m):
    if m == 1:
        return f
    else:
        return Pow_Impl(f, m)
