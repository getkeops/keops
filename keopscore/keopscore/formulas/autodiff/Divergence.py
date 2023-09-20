from keopscore.formulas import Var
from keopscore.formulas.LinearOperators import TraceOperator
from keopscore.utils.code_gen_utils import GetInds
from keopscore.formulas.variables.IntCst import IntCst
from keopscore.utils.misc_utils import KeOps_Error

# /////////////////////////////////////////////////////////////
# ///      DIVERGENCE OPERATOR  : Divergence< F, V, U >       ////
# /////////////////////////////////////////////////////////////

# Defines Div_v(F), the divergence of F with respect to v
# F and v must have same dimension.


def Divergence(formula, v):
    if formula.dim != v.dim:
        KeOps_Error(
            "Divergence requires formula and variable to have the same dimaension."
        )

    # we define a temporary gradin to backpropagate.
    # This gradin will disappear in the formula when we compute the trace.
    # We define this temporary gradin as Var(ind,dim,cat)
    # with ind : a new index that must not exist in the formula,
    # dim : must equal formula.dim = v.dim
    # cat : not specified and not used, we put -1
    inds = GetInds(formula.Vars_)
    ind = 1 + max(inds) if len(inds) > 0 else 0
    dim = v.dim
    cat = -1
    gradin = Var(ind, dim, cat)

    Gf = formula.DiffT(v, gradin)

    return TraceOperator(Gf, gradin)
