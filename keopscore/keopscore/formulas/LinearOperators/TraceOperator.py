from .LinearOperator import LinearOperator_class
from .SumLinOperator import SumLinOperator

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
# ///      TRACE OPERATOR       ////
# /////////////////////////////////////////////////////////////


class TraceOperator_class(LinearOperator_class):
    string_id = "TraceOperator"

    def call(self, formula, v, w=1):
        """
        Trace of a linear function.
        formula must be a linear function of v
        If formula.dim = v.dim:
            If w=1, computes the trace of formula with respect to v,
                i.e. tr(M) = sum_i M_ii if M is the matrix representation of formula
            If w.dim=1, computes w * tr(M)
            If w.dim=v.dim, computes a weighted trace, i.e. tr(diag(w)xM) = sum_i w_i M_ii
        If formula.dim=1: then the matrix representation of the linear function is M of size (1,d) with d=v.dim
            If w=1, computes the trace of 1xM with 1 of size (d,1)
                i.e. tr(1xM) = sum_i M_i
            If w.dim=1, computes w * tr(1xM) = w * sum_i M_i
            If w.dim=v.dim, computes a weighted trace, i.e. tr(diag(w)x1xM) = sum_i w_i M_i
        """

        wdim = 1 if w == 1 else w.dim

        if not wdim in [1, v.dim]:
            KeOps_Error("Dimension of argument w should be 1 or equal dimension of v.")

        if formula.dim not in [1, v.dim]:
            KeOps_Error(
                f"Output dimension should be 1 or {v.dim} ({formula.string_id}, dim {formula.dim})."
            )

        if isinstance(formula, Mult_Impl):
            fa, fb = formula.children
            if fa.is_linear(v):
                # If formula.dim=v.dim=d:
                #   If fb.dim=1, the matrix of fa*fb is fb*Ma
                #       if w.dim=1, w*tr(fb*Ma) = w*fb*tr(Ma)
                #       if w.dim=d, tr(diag(w)x(fb*Ma)) = tr(diag(fb*w)xMa)
                #   If fb.dim=v.dim, the matrix of fa*fb is diag(fb)xMa
                #       if w.dim=1, w*tr(diag(fb)xMa) = tr(diag(w*fb)xMa)
                #       if w.dim=d, tr(diag(w)xdiag(fb)xMa) = tr(diag(fb*w)xMa)
                # If formula.dim=1:
                #   If fb.dim=1, the matrix of fa*fb is fb*Ma
                #       if w.dim=1, w*tr(1x(fb*Ma)) = w*fb*tr(1xMa)
                #       if w.dim=d, tr(diag(w)x1x(fb*Ma)) = tr(diag(fb*w)x1xMa)
                #   If fb.dim=v.dim, the matrix of fa*fb is diag(fb)xMa
                #       if w.dim=1, w*tr(diag(fb)x1xMa) = tr(diag(w*fb)x1xMa)
                #       if w.dim=d, tr(diag(w)xdiag(fb)x1xMa) = tr(diag(fb*w)x1xMa)
                if fb.dim == 1:
                    # just better for formula simplification
                    return self(fa, v, w) * fb
                return self(fa, v, fb * w)
            elif fb.is_linear(v):
                return self(fb, v, fa * w)
        elif isinstance(formula, Divide_Impl):
            fa, fb = formula.children
            return self(fa, v, w / fb)
        elif isinstance(formula, Var):
            # Here we must have formula=v, so the matrix is Identity : M=I
            # If w.dim=1, w*tr(I) = w*d
            # If w.dim=d, tr(diag(w)xI) = Sum(w)
            return IntCst_Impl(v.dim) * w if wdim == 1 else Sum(w)
        elif isinstance(formula, SumT_Impl):
            (f,) = formula.children
            # here f.dim=1, and the matrix of SumT(f) is 1xMa with 1 of size (d,1) and Ma of size (1,d)
            # So by convention we just compute the same operator on f
            return self(f, v, w)
        elif isinstance(formula, Sum_Impl):
            (f,) = formula.children
            # here formula.dim=1
            # and the matrix of Sum(f) is 1' x Ma with 1' of size (1,m) and Ma of size (m,d)
            #   If w.dim=1, w*tr(1x1'xMa) = w*sum(Ma)
            #   If w.dim=d, tr(diag(w)x1x1'xMa) = sum_ij w_j M_ij
            return SumLinOperator(f, v, wr=w)


def TraceOperator(formula, v, w=1):
    return TraceOperator_class()(formula, v, w=w)
