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
# ///      LINEAR OPERATORS       ////
# /////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////
# ///      BASE CLASS       ////
# /////////////////////////////////////////////////////////////


class LinearOperator_class:
    def __call__(self, formula, v, *args, **kwargs):
        m, n = formula.dim, v.dim
        if not formula.is_linear(v):
            KeOps_Error("Formula is not linear with respect to variable.")

        if type(formula) in [Add_Impl, Minus_Impl, Subtract_Impl]:
            newargs = (
                self(Broadcast(f, m), v, *args, **kwargs) for f in formula.children
            )
            return type(formula)(*newargs, *formula.params)
        elif isinstance(formula, Scalprod_Impl):
            fa, fb = formula.children
            return self(Sum_Impl(fa * fb), v, *args, **kwargs)

        res = self.call(formula, v, *args, **kwargs)
        if res is not None:
            return res
        else:
            KeOps_Error(f"{self.string_id} not implemented for {formula.string_id}")
