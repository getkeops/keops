from keopscore.formulas.Operation import Broadcast
from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.maths.Mult import Mult_Impl
from keopscore.formulas.maths.Minus import Minus_Impl
from keopscore.formulas.maths.Sum import Sum
from keopscore.formulas.variables.IntCst import IntCst, IntCst_Impl
from keopscore.formulas.variables.RatCst import RatCst, RatCst_Impl
from keopscore.formulas.variables.Zero import Zero


##########################
######    Add        #####
##########################


class Add_Impl(VectorizedScalarOp):
    """the binary addition operation"""

    string_id = "Add"
    print_spec = "+", "mid", 4
    linearity_type = "all"

    def ScalarOp(self, out, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {arg0.id}+{arg1.id};\n"

    def DiffT(self, v, gradin):
        fa, fb = self.children
        if fa.dim == 1 and fb.dim > 1:
            return fa.DiffT(v, Sum(gradin)) + fb.DiffT(v, gradin)
        elif fb.dim == 1 and fa.dim > 1:
            return fa.DiffT(v, gradin) + fb.DiffT(v, Sum(gradin))
        else:
            return fa.DiffT(v, gradin) + fb.DiffT(v, gradin)

    # parameters for testing the operation (optional)
    nargs = 2  # number of arguments


# N.B. The following separate function could theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Add(arg0, arg1):
    from keopscore.formulas.maths.Subtract import Subtract_Impl

    # Simplification rules

    # x+0=x, 0+x=x, x+x=2x
    if isinstance(arg0, Zero):
        return Broadcast(arg1, arg0.dim)
    elif isinstance(arg1, Zero):
        return Broadcast(arg0, arg1.dim)
    elif arg0 == arg1:
        return IntCst(2) * arg0

    # m+n with m,n integers or rationals
    if isinstance(arg0, IntCst_Impl):
        if isinstance(arg1, IntCst_Impl):
            # m+n
            return IntCst(arg0.val + arg1.val)
        if isinstance(arg1, RatCst_Impl):
            # m+p/q -> (mq+p)/q
            return RatCst(arg0.val * arg1.q + arg1.p, arg1.q)
    if isinstance(arg0, RatCst_Impl):
        if isinstance(arg1, IntCst_Impl):
            # p/q+n -> (p+nq)/q
            return RatCst(arg0.p + arg1.val * arg0.q, arg1.q)
        if isinstance(arg1, RatCst_Impl):
            # (p/q)+(m/n) -> (pn+qm)/(qn)
            return RatCst(arg0.p * arg1.q + arg0.q * arg1.p, arg0.q * arg1.q)

    # gathering integers or rationals in e.g. m+(n+x) = (m+n)+x
    if isinstance(arg0, (IntCst_Impl, RatCst_Impl)):
        if isinstance(arg1, Add_Impl) and isinstance(
            arg1.children[0], (IntCst_Impl, RatCst_Impl)
        ):
            # m+(r+g) -> (m+r)+g
            return (arg0 + arg1.children[0]) + arg1.children[1]
        if isinstance(arg1, Subtract_Impl) and isinstance(
            arg1.children[0], (IntCst_Impl, RatCst_Impl)
        ):
            # m+(r-g) -> (m+r)-g
            return (arg0 + arg1.children[0]) - arg1.children[1]
    if isinstance(arg1, (IntCst_Impl, RatCst_Impl)):
        if isinstance(arg0, Add_Impl) and isinstance(
            arg0.children[0], (IntCst_Impl, RatCst_Impl)
        ):
            # (r+g)+m -> (m+r)+g
            return (arg1 + arg0.children[0]) + arg0.children[1]
        if isinstance(arg0, Subtract_Impl) and isinstance(
            arg0.children[0], (IntCst_Impl, RatCst_Impl)
        ):
            # (r-g)+m -> (m+r)-g
            return (arg1 + arg0.children[0]) - arg0.children[1]

    # rules for factorization ax+bx=(a+b)x etc.
    if isinstance(arg0, Mult_Impl) and isinstance(arg1, Mult_Impl):
        # ab+cd
        a, b, c, d = *arg0.children, *arg1.children
        if a == c:
            return a * (b + d)
        if a == d:
            return a * (b + c)
        if b == c:
            return b * (a + d)
        if b == d:
            return b * (a + c)
    if (
        isinstance(arg0, Mult_Impl)
        and isinstance(arg0.children[0], IntCst_Impl)
        and arg0.children[1] == arg1
    ):
        #  factorization :  n*x + x = (n+1)*x
        return IntCst(arg0.children[0].val + 1) * arg1
    if (
        isinstance(arg1, Mult_Impl)
        and isinstance(arg1.children[0], IntCst_Impl)
        and arg1.children[1] == arg0
    ):
        #  factorization :  x + n*x = (n+1)*x
        return IntCst(arg1.children[0].val + 1) * arg0
    if isinstance(arg0, Minus_Impl) and isinstance(arg1, Minus_Impl):
        # (-x)+(-y) -> -(x+y)
        return -(arg0.children[0] + arg1.children[0])
    if isinstance(arg1, Minus_Impl):
        return arg0 - arg1.children[0]
    return Add_Impl(arg0, arg1)
