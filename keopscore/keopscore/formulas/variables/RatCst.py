from keopscore.utils.code_gen_utils import cast_to, c_variable
from keopscore.formulas.Operation import Operation
from keopscore.formulas.variables.Zero import Zero
from keopscore.formulas.variables.IntCst import IntCst
from fractions import Fraction


class RatCst_Impl(Operation):
    # constant rational number "operation"
    string_id = "RatCst"

    def recursive_str(self):
        return f"{self.p}/{self.q}"

    def __init__(self, p, q):
        super().__init__(params=(p, q))
        self.p = p
        self.q = q
        self.dim = 1

    def Op(self, out, table):
        float_val = c_variable("float", f"(float)({self.p/self.q})")
        return f"*{out.id} = {cast_to(out.dtype, float_val)};\n"

    def DiffT(self, v, gradin):
        return Zero(v.dim)


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def RatCst(a, b):
    r = Fraction(a, b)
    p, q = r.numerator, r.denominator
    if p == 0:
        return Zero(1)
    elif q == 1:
        return IntCst(p)
    else:
        return RatCst_Impl(p, q)
