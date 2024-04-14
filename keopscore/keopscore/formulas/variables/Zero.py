from keopscore.utils.code_gen_utils import c_zero_float
from keopscore.formulas.Operation import Operation


class Zero(Operation):
    """zero operation : encodes a vector of zeros"""

    string_id = "Zero"

    def is_linear(self, v):
        return True

    def __init__(self, dim=None):
        super().__init__(params=(dim,))
        self.dim = dim

    # custom __eq__ method
    def __eq__(self, other):
        return type(self) == type(other) and self.dim == other.dim

    def Op(self, out, table):
        return out.assign(c_zero_float)

    def DiffT(self, v, gradin):
        return Zero(v.dim)
