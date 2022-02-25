from keopscore.utils.code_gen_utils import c_zero_float
from keopscore.formulas.Operation import Operation


class Zero(Operation):
    """zero operation : encodes a vector of zeros"""

    string_id = "Zero"

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.params = (dim,)

    # custom __eq__ method
    def __eq__(self, other):
        return type(self) == type(other) and self.dim == other.dim

    def Op(self, out, table):
        return out.assign(c_zero_float)

    def DiffT(self, v, gradin):
        return Zero(v.dim)
