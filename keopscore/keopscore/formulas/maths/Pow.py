from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_pow


class Pow_Impl(VectorizedScalarOp):

    # parameters for testing the operation (optional)
    nargs = 1  # number of arguments (excluding parameters)
    test_ranges = [(0, 2)]  # ranges of arguments
    test_params = [2]  # values of parameters for testing
    torch_op = "lambda x,m : torch.pow(x, m)"


class Pow_Factory:

    def __init__(self, m):

        class Class(Pow_Impl):
            """the integer power vectorized operation
            Pow(f,m) where m is integer, computes f^m
            """

            string_id = "Pow"
            print_fun = lambda x: f"{x}**{m}"
            print_level = 4

            ScalarOpFun = lambda x: keops_pow(x, m)

            @staticmethod
            def Derivative(f):
                from keopscore.formulas.variables.IntCst import IntCst

                return IntCst(m) * Pow(f, m - 1)

        self.Class = Class

    def __call__(self, f):

        return self.Class(f)


# N.B. The following separate function could theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Pow(f, m):
    if m == 1:
        return f
    else:
        return Pow_Factory(m)(f)
