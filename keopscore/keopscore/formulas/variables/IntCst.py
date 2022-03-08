from keopscore.utils.code_gen_utils import cast_to, c_variable
from keopscore.formulas.Operation import Operation
from keopscore.formulas.variables.Zero import Zero


class IntCst_Impl(Operation):
    # constant integer "operation"
    string_id = "IntCst"
    print_spec = "", "pre", 0

    def __init__(self, val):
        super().__init__()
        self.val = val
        self.dim = 1
        self.params = (val,)

    # custom __eq__ method
    def __eq__(self, other):
        return type(self) == type(other) and self.val == other.val

    def Op(self, out, table):
        float_val = c_variable("float", f"(float){self.val}")
        return f"*{out.id} = {cast_to(out.dtype, float_val)};\n"

    def DiffT(self, v, gradin):
        return Zero(v.dim)


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def IntCst(arg):
    if arg == 0:
        return Zero(1)
    else:
        return IntCst_Impl(arg)
