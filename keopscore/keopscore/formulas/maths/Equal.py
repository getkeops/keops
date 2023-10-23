from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_equal
from keopscore.formulas.variables.Zero import Zero


class Equal(VectorizedScalarOp):
    """the "equal" vectorized operation
    Equal(f,g) = 1 if f==g, 0 otherwise
    """

    string_id = "Equal"
    print_spec = "==", "mid", 3

    ScalarOpFun = keops_equal

    def DiffT(self, v, gradin):
        return Zero(v.dim)

    # parameters for testing the operation (optional)
    nargs = 2  # number of arguments
    torch_op = None  # "lambda x,y : torch.eq(x,y).float()"
