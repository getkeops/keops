from keopscore.utils.meta_toolbox.c_expression import c_expression
from keopscore.utils.code_gen_utils import cast_to
from keopscore.formulas.Operation import Operation
from keopscore.formulas.variables.Zero import Zero


class IntCst_Impl(Operation):
    # constant integer "operation"
    string_id = "IntCst"
    print_spec = "", "pre", 0

    def __init__(self, val=None, params=None):
        # N.B. init via params keyword is used for compatibility with base class.
        if val is None:
            # here params should be a tuple containing one single integer
            (val,) = params
        super().__init__(params=(val,))
        self.val = val
        self.dim = 1

    def Op(self, out, table):
        dtype = "int" if abs(self.val) < 2e9 else "signed long int"
        c_val = c_expression(str(self.val), set(), dtype)
        return out.assign(c_val)

    def DiffT(self, v, gradin):
        return Zero(v.dim)


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def IntCst(arg):
    if arg == 0:
        return Zero(1)
    else:
        return IntCst_Impl(arg)
