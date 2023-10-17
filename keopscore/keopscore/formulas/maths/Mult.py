from keopscore.formulas.Operation import Broadcast
from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.maths.Exp import Exp
from keopscore.formulas.maths.Scalprod import Scalprod
from keopscore.formulas.maths.Square import Square
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.math_functions import keops_mul
from keopscore.formulas.variables.IntCst import IntCst_Impl, IntCst
from keopscore.formulas.variables.RatCst import RatCst_Impl, RatCst
from keopscore.formulas.maths.SumT import SumT, SumT_Impl
from keopscore.utils.misc_utils import KeOps_Error

##########################
######    Mult       #####
##########################


class Mult_Impl(VectorizedScalarOp):
    """the binary multiply operation"""

    string_id = "Mult"
    print_spec = "*", "mid", 3
    linearity_type = "one"

    ScalarOpFun = keops_mul

    #  \diff_V (A*B) = (\diff_V A) * B + A * (\diff_V B)
    def DiffT(self, v, gradin):
        fa, fb = self.children
        if isinstance(fb, Exp) and fb.dim == 1:
            (u,) = fb.children
            return (fa.DiffT(v, gradin) + u.DiffT(v, Scalprod(gradin, fa))) * fb
        if fa.dim == 1 and fb.dim > 1:
            return fa.DiffT(v, Scalprod(gradin, fb)) + fb.DiffT(v, fa * gradin)
        elif fb.dim == 1 and fa.dim > 1:
            return fa.DiffT(v, fb * gradin) + fb.DiffT(v, Scalprod(gradin, fa))
        else:
            return fa.DiffT(v, fb * gradin) + fb.DiffT(v, fa * gradin)

    # parameters for testing the operation (optional)
    nargs = 2  # number of arguments
    torch_op = "torch.mul"  # equivalent PyTorch operation


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Mult(arg0, arg1):
    from keopscore.formulas.maths.Minus import Minus_Impl

    if isinstance(arg0, Zero):
        return Broadcast(arg0, arg1.dim)
    elif isinstance(arg1, Zero):
        return Broadcast(arg1, arg0.dim)
    elif isinstance(arg0, IntCst_Impl):
        if arg0.val == 1:
            # 1*f -> f
            return arg1
        if arg0.val == -1:
            # -1*f -> -f
            return -arg1
        elif isinstance(arg1, IntCst_Impl):
            # m*n -> mn
            return IntCst_Impl(arg0.val * arg1.val)
        elif isinstance(arg1, RatCst_Impl):
            # m*(p/q) -> (m*p)/q
            return RatCst(arg0.val * arg1.p, arg1.q)
    if isinstance(arg1, IntCst_Impl):
        # f*n -> n*f (bringing integers to the left)
        return Mult(arg1, arg0)
    if isinstance(arg0, RatCst_Impl) and isinstance(arg1, RatCst_Impl):
        # (p/q)*(m/n) -> (pm)/(qn)
        return RatCst(arg0.p * arg1.p, arg0.q * arg1.q)
    elif isinstance(arg1, Mult_Impl) and isinstance(arg1.children[0], IntCst_Impl):
        if isinstance(arg0, IntCst_Impl):
            # m*(n*g) -> (m*n)*g
            return (
                IntCst(arg0.params[0] * arg1.children[0].params[0]) * arg1.children[1]
            )
        # f*(n*g) -> n*(f*g)
        return arg1.children[0] * (arg0 * arg1.children[1])
    elif isinstance(arg1, Mult_Impl) and isinstance(arg1.children[0], RatCst_Impl):
        if isinstance(arg0, (IntCst_Impl, RatCst_Impl)):
            # m*(r*g) -> (m*r)*g
            m, r, g = arg0, *arg1.children
            return (m * r) * g
        # f*(n*g) -> n*(f*g)
        return arg1.children[0] * (arg0 * arg1.children[1])
    elif isinstance(arg0, Mult_Impl) and isinstance(
        arg0.children[0], (IntCst_Impl, RatCst_Impl)
    ):
        if not isinstance(arg1, Exp):
            # (n*f)*g -> n*(f*g)
            return arg0.children[0] * (arg0.children[1] * arg1)
    elif isinstance(arg0, Minus_Impl) and isinstance(arg1, Minus_Impl):
        # (-f)*(-g) -> f*g
        return arg0.children[0] * arg1.children[0]
    elif isinstance(arg0, Minus_Impl):
        # (-f)*g -> -(f*g)
        return -(arg0.children[0] * arg1)
    elif isinstance(arg1, Minus_Impl):
        # f*(-g) -> -(f*g)
        return -(arg0 * arg1.children[0])
    elif arg0 == arg1:
        # f*f -> f^2
        return Square(arg0)
    elif isinstance(arg1, SumT_Impl):
        # f*SumT(g)
        if isinstance(arg0, SumT_Impl):
            # SumT(f)*SumT(g) -> SumT(f*g)
            if arg0.dim != arg1.dim:  # should never happen...
                KeOps_Error("dimensions are not compatible for Mult operation")
            return SumT(arg0.children[0] * arg1.children[0], arg0.dim)
        elif arg0.dim == 1:
            # f*SumT(g) -> SumT(f*g)
            return SumT(arg0 * arg1.children[0], arg1.dim)
        elif arg1.dim == arg0.dim:
            # f*SumT(g) -> f*g (broadcasted)
            return arg0 * arg1.children[0]
        else:
            KeOps_Error("dimensions are not compatible for Mult operation")
    elif isinstance(arg0, SumT_Impl):
        # SumT(f)*g -> g*SumT(f)
        return arg1 * arg0
    elif isinstance(arg0, Exp) and isinstance(arg1, Exp):
        # Exp(f)*Exp(g) -> Exp(f+g)
        return Exp(arg0.children[0] + arg1.children[0])
    elif isinstance(arg0, Exp) and arg0.dim >= arg1.dim:
        # Exp(f)*g -> g*Exp(f) except if g.dim > f.dim
        # This is to bring exponentials to the right for further simplifications
        return arg1 * arg0
    elif isinstance(arg0, Mult_Impl):
        u, v = arg0.children
        # (u*v)*g
        if isinstance(v, Exp):
            (a,) = v.children
            # (u*Exp(a))*g
            if isinstance(arg1, Exp):
                (b,) = arg1.children
                # (u*Exp(a))*Exp(b) -> u*Exp(a+b)
                return u * Exp(a + b)
            elif arg1.dim <= a.dim:
                # (u*Exp(a))*g -> (u*g)*Exp(a) except if g.dim > a.dim
                return (u * arg1) * v
    elif (
        isinstance(arg1, Mult_Impl)
        and isinstance(arg1.children[1], Exp)
        and not isinstance(arg0, IntCst_Impl)
    ):
        # f*(u*Exp(v)) -> (f*u)*Exp(v)
        return (arg0 * arg1.children[0]) * arg1.children[1]
    return Mult_Impl(arg0, arg1)
