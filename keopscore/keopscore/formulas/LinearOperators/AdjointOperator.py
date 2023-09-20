from .LinearOperator import LinearOperator_class

from keopscore.formulas import Var
from keopscore.utils.code_gen_utils import GetInds
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.formulas.maths.Minus import Minus_Impl
from keopscore.formulas.maths.Add import Add_Impl
from keopscore.formulas.maths.Subtract import Subtract_Impl
from keopscore.formulas.maths.Mult import Mult_Impl
from keopscore.formulas.maths.Sum import Sum_Impl, Sum
from keopscore.formulas.maths.SumT import SumT_Impl, SumT
from keopscore.formulas.maths.Divide import Divide_Impl
from keopscore.formulas.maths.Exp import Exp
from keopscore.formulas.variables.IntCst import IntCst_Impl
from keopscore.formulas.variables.Var import Var
from keopscore.formulas.variables.Zero import Zero
from keopscore.formulas.maths.Scalprod import Scalprod_Impl
from keopscore.formulas.Operation import Broadcast


# /////////////////////////////////////////////////////////////
# ///      ADJOINT OPERATOR       ////
# /////////////////////////////////////////////////////////////


class AdjointOperator_class(LinearOperator_class):
    string_id = "AdjointOperator"

    def call(self, formula, v, u):
        m, n = formula.dim, v.dim
        if isinstance(formula, Mult_Impl):
            fa, fb = formula.children
            if fa.is_linear(v):
                fa = Broadcast(fa, m)
                return self(fa, v, fb * u)
            elif fb.is_linear(v):
                fb = Broadcast(fb, m)
                return self(fb, v, fa * u)
        elif isinstance(formula, Divide_Impl):
            fa, fb = formula.children
            fa = Broadcast(fa, m)
            return self(fa, v, u / fb)
        elif isinstance(formula, Var):
            return u  # N.B. we must have v==formula here
        elif isinstance(formula, SumT_Impl):
            (f,) = formula.children
            return self(f, v, Sum(u))
        elif isinstance(formula, Sum_Impl):
            (f,) = formula.children
            return self(f, v, SumT(u, f.dim))


def AdjointOperator(formula, v, u):
    return AdjointOperator_class()(formula, v, u)
