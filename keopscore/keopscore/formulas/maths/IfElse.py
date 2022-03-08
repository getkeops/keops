from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.math_functions import keops_ifelse


class IfElse(VectorizedScalarOp):
    """the if/else vectorized operation
    IfElse(f,a,b) = a if f>=0, b otherwise
    """

    string_id = "IfElse"

    ScalarOpFun = keops_ifelse

    def DiffT(self, v, gradin):
        f, a, b = self.children
        return IfElse(f, a.DiffT(v, gradin), b.DiffT(v, gradin))

    # parameters for testing the operation (optional)
    nargs = 3  # number of arguments
