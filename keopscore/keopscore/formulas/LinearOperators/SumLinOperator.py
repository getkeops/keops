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
# ///      SUM_LIN OPERATOR       ////
# /////////////////////////////////////////////////////////////


class SumLinOperator_class(LinearOperator_class):
    string_id = "SumLinOperator"

    def call(self, formula, v, wl=1, wr=1):
        """
        Sum of all matrix entries of a linear function.
        formula must be a linear function of v
        If wr=1,
            If wl=1, computes sum(M) = sum_ij M_ij if M is the matrix representation of formula
            If wl.dim=1, computes sum( wl x M ) = wl * sum(M) = sum_ij wl M_ij
            If wl.dim=formula.dim, computes a weighted sum, i.e. sum(diag(wl)xM) = sum_ij wl_i M_ij
        If wr.dim=1,
            If wl=1, computes sum( M x wr ) = sum_ij wr M_ij
            If wl.dim=1, computes sum( wl x M x wr ) = sum_ij wl wr M_ij
            If wl.dim=formula.dim, computes a weighted sum, i.e. sum(diag(wl) x M x wr) = wr * sum_ij w_i M_ij
        If wr.dim=v.dim, computes a weighted sum,
            If wl=1, computes sum(Mxdiag(wr)) = sum_ij wr_j M_ij if M is the matrix representation of formula
            If wl.dim=1, computes wl * sum(Mxdiag(wr))
            If wl.dim=formula.dim, computes sum(diag(wl)xMxdiag(wr)) = sum_ij wl_i M_ij wr_j
        """

        wldim = 1 if wl == 1 else wl.dim
        wrdim = 1 if wr == 1 else wr.dim

        assert wldim in [1, formula.dim]
        assert wrdim in [1, v.dim]

        if isinstance(formula, Mult_Impl):
            fa, fb = formula.children
            if fa.is_linear(v):
                fa = Broadcast(fa, formula.dim)
                # If fb.dim=1, the matrix of fa*fb is M=fb*Ma
                #   if wr.dim=1,
                #       if wl.dim=1, wl*wr*sum(fb*Ma) = sum((wl*fb) x Ma x wr)
                #       If wl.dim=formula.dim, wr*sum(diag(wl)x(fb*Ma)) = sum(diag(fb*wl) x Ma x wr)
                #   if wr.dim=v.dim,
                #       if wl.dim=1, wl*sum(fb*Ma x diag(wr)) = sum( wl*fb x Ma x diag(wr))
                #       If wl.dim=formula.dim, sum(diag(wl) x (fb*Ma) x diag(wr)) = sum(diag(fb*wl) x Ma x diag(wr))
                # If fb.dim=formula.dim, the matrix of fa*fb is diag(fb)xMa
                #   if wr.dim=1,
                #       if wl.dim=1, wl*wr*sum(diag(fb)xMa) = sum(diag(wl*fb) x Ma x wr)
                #       If wl.dim=formula.dim, wr*sum(diag(wl)x(fb*Ma)) = wr*sum(diag(fb*wl)xMa)
                #   if wr.dim=v.dim,
                #       if wl.dim=1, wl*sum(fb*Ma x diag(wr)) = wl*fb*sum(Ma x diag(wr))
                #       If wl.dim=formula.dim, sum(diag(wl) x (fb*Ma) x diag(wr)) = sum(diag(fb*wl) x Ma x diag(wr))
                return self(fa, v, fb * wl, wr)
            elif fb.is_linear(v):
                fb = Broadcast(fb, formula.dim)
                return self(fb, v, fa * wl, wr)
        elif isinstance(formula, Divide_Impl):
            fa, fb = formula.children
            fa = Broadcast(fa, formula.dim)
            return self(fa, v, wl / fb, wr)
        elif isinstance(formula, Var):
            # Here we must have formula=v, so the matrix is Identity : M=I
            # If wr.dim=1,
            #   If wl.dim=1, wl*wr*sum(I) = wl*wr*d
            #   If wl.dim=d, wr*sum(diag(wl)xI) = Sum(wl)*wr
            # If wr.dim=v.dim,
            #   If wl.dim=1, wl*sum(Ixdiag(wr)) = wl * Sum(wr)
            #   If wl.dim=d, sum(diag(wl) x I x diag(wr)) = Sum(wl*wr)
            return (
                IntCst_Impl(v.dim) * wl * wr
                if (wldim == 1 and wrdim == 1)
                else Sum(wl * wr)
            )
        elif isinstance(formula, SumT_Impl):
            (f,) = formula.children
            # here f.dim=1, and the matrix of SumT(f) is 1xMa with 1 of size (n,1) and Ma of size (1,d)
            # If wr.dim=1,
            #   If wl.dim=1, wl*wr*sum(1xMa) = wl*n*sum(Ma)*wr
            #   If wl.dim=n, wr*sum(diag(wl)x1xMa) = Sum(wl)*sum(Ma)*wr
            # If wr.dim=d,
            #   If wl.dim=1, wl*sum(1 x Ma x diag(wr)) = wl*n*sum(Ma x diag(wr))
            #   If wl.dim=n, sum(diag(wl) x 1 x Ma x diag(wr)) = Sum(wl)*sum(Ma  x diag(wr))
            wl = wl * formula.dim if wldim == 1 else Sum(wl)
            return wl * self(f, v, wr=wr)
        elif isinstance(formula, Sum_Impl):
            (f,) = formula.children
            # here formula.dim=1, wl.dim=1,
            # and the matrix of Sum(f) is 1' x Ma with 1' of size (1,m) and Ma of size (m,d)
            # If wr.dim=1, wl*wr*sum(1'xMa) = wl*wr*sum(Ma)
            # If wr.dim=d, wl*sum(1' x Ma x diag(wr)) = wl*sum(Ma x diag(wr))
            return wl * self(f, v, wr)


def SumLinOperator(formula, v, wl=1, wr=1):
    return SumLinOperator_class()(formula, v, wl=wl, wr=wr)
