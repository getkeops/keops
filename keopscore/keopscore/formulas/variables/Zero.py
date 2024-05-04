from keopscore.utils.meta_toolbox import c_zero_float
from keopscore.formulas.Operation import Operation
from keopscore.utils.unique_object import unique_object


def Zero(dim):
    return Zero_Impl_Factory(dim)()


class Zero_Impl(Operation):
    pass


class Zero_Impl_Factory(metaclass=unique_object):

    def __init__(self, dim):

        class Class(Zero_Impl):
            """zero operation : encodes a vector of zeros"""

            string_id = f"Zero({dim})"
            print_fun = lambda: Class.string_id

            def is_linear(self, v):
                return True

            def __init__(self):

                super().__init__()
                self.dim = dim

            def Op(self, out, table):
                return out.assign(c_zero_float)

            def DiffT(self, v, gradin):
                return Zero(v.dim)

        self.Class = Class

    def __call__(self):
        return self.Class()
