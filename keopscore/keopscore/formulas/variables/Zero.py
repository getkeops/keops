from keopscore.utils.meta_toolbox import c_zero_float
from keopscore.formulas.Operation import Operation


def Zero(dim):
    return Zero_Impl_Factory(dim)()


class Zero_Impl(Operation):
    pass


class Zero_Impl_Factory:

    def __init__(self, dim):

        class Class(Zero_Impl):
            """zero operation : encodes a vector of zeros"""

            string_id = "Zero"

            def is_linear(self, v):
                return True

            def __init__(self):

                super().__init__()
                self.dim = dim

            # custom __eq__ method
            def __eq__(self, other):
                return type(self) == type(other) and self.dim == other.dim

            def Op(self, out, table):
                return out.assign(c_zero_float)

            def DiffT(self, v, gradin):
                return Zero(v.dim)

        self.Class = Class

    def __call__(self):
        return self.Class()
