from keops.python_engine.code_gen_utils import c_variable
from keops.python_engine.formulas.Operation import Operation


class Zero(Operation):
    # zero operation : encodes a vector of zeros
    string_id = "Zero"

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.params = (dim,)

    # custom __eq__ method
    def __eq__(self, other):
        return type(self) == type(other) and self.dim == other.dim

    def Op(self, out, table):
        zero = c_variable("float", "0.0f")
        return out.assign(zero)

    def DiffT(self, v, gradin):
        return Zero(v.dim)