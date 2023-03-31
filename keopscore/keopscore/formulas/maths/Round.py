from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.math_functions import keops_round


class Round(VectorizedScalarOp):
    """the Round vectorized operation
    Round(f,d) where d is integer, rounds f to d decimals
    """

    def __init__(self, f, d):
        super().__init__(f, params=(d,))

    string_id = "Round"

    ScalarOpFun = keops_round

    def DiffT(self, v, gradin):
        return Zero(v.dim)

    # parameters for testing the operation (optional)
    nargs = 1  # number of arguments
    test_params = [3]  # parameters to try
    torch_op = None  # equivalent PyTorch operation
