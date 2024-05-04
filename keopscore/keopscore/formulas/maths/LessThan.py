from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_lessthan
from keopscore.formulas.variables.Zero import Zero


class LessThan(VectorizedScalarOp):
    """the less-than vectorized operation
    LessThan(f,g) = 1 if f<g, 0 otherwise
    """

    string_id = "LessThan"
    print_fun = lambda x, y: f"{x}<{y}"
    print_level = 3

    ScalarOpFun = keops_lessthan

    def GradFun(self, v, gradin):
        return Zero(v.dim)

    # parameters for testing the operation (optional)
    nargs = 2  # number of arguments
