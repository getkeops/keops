from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_notequal
from keopscore.formulas.variables.Zero import Zero


class NotEqual(VectorizedScalarOp):
    """the "not-equal" vectorized operation
    NotEqual(f,g) = 1 if f!=g, 0 otherwise
    """

    string_id = "NotEqual"
    print_spec = "!=", "mid", 3

    ScalarOpFun = keops_notequal

    def DiffT(self, v, gradin):
        return Zero(v.dim)

    # parameters for testing the operation (optional)
    nargs = 2  # number of arguments
    torch_op = "lambda x,y : torch.ne(x,y).float()"
    no_torch_grad = True
