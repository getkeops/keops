from keopscore.utils.meta_toolbox import c_empty_instruction, c_array_scalar
from keopscore.formulas.Operation import Operation
from keopscore.formulas.variables.Zero import Zero


class IntCst_Impl(Operation):
    pass


class IntCst_Impl_Factory:

    def __init__(self, val):

        class Class(IntCst_Impl):

            # constant integer "operation"
            string_id = str(val)
            print_fun = lambda: Class.string_id

            def __init__(self):
                super().__init__()
                self.dim = 1
                self.val = val

            def get_code_and_expr(self, dtype, table, i, j, tagI):
                return c_empty_instruction, c_array_scalar(val)

            def DiffT(self, v, gradin):
                return Zero(v.dim)

        self.Class = Class

    def __call__(self):
        return self.Class()


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def IntCst(arg):
    if arg == 0:
        return Zero(1)
    else:
        return IntCst_Impl_Factory(arg)()
