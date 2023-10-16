from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_lessorequal
from keopscore.formulas.variables.Zero import Zero


class LessOrEqual(VectorizedScalarOp):
    """the less-than-or-equal-to vectorized operation
    LessOrEqual(f,g) = 1 if f<=g, 0 otherwise
    """

    string_id = "LessOrEqual"
    print_spec = "<=", "mid", 3

    ScalarOpFun = keops_lessorequal

    def DiffT(self, v, gradin):
        return Zero(v.dim)

    # parameters for testing the operation (optional)
    nargs = 2  # number of arguments
