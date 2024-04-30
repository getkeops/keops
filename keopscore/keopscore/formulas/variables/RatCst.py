from keopscore.utils.meta_toolbox import (
    c_empty_instruction,
    c_expression,
    c_array_scalar,
)
from keopscore.formulas.Operation import Operation
from keopscore.formulas.variables.Zero import Zero
from keopscore.formulas.variables.IntCst import IntCst
from fractions import Fraction


class RatCst_Impl(Operation):
    pass


class RatCst_Impl_Factory:

    def __init__(self, p, q):

        class Class(RatCst_Impl):

            # constant rational number "operation"
            string_id = "RatCst"

            def recursive_str(self):
                return f"{self.p}/{self.q}"

            def __init__(self):
                super().__init__()
                self.p = p
                self.q = q
                self.dim = 1

            def get_code_and_expr(self, dtype, table, i, j, tagI):
                return c_empty_instruction, c_array_scalar(self.p / self.q)

            def DiffT(self, v, gradin):
                return Zero(v.dim)

        self.Class = Class

    def __call__(self):
        return self.Class()


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
        return RatCst_Impl_Factory(p, q)()
