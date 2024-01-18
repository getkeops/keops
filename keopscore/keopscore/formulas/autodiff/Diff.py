from keopscore.formulas import Var, Reduction
from keopscore.formulas.LinearOperators import AdjointOperator
from keopscore.utils.code_gen_utils import GetInds

# /////////////////////////////////////////////////////////////
# ///      DIFF OPERATOR  : Diff< F, V, U >       ////
# /////////////////////////////////////////////////////////////

# Defines D_v(F).u, the differential of F with respect to v, applied to u
# D_v(F) is the adjoint of the Gradient


def Diff(formula, v, u):
    if isinstance(formula, Reduction):
        return formula.Diff(v, u)
    # we define a temporary gradin to backpropagate.
    # This gradin will disappear in the formula when we compute the adjoint.
    # We define this temporary gradin as Var(ind,dim,cat)
    # with ind : a new index that must not exist in the formula,
    # dim : must equal formula.dim
    # cat : not specified and not used, we put -1
    if isinstance(v, tuple):
        return sum(Diff(formula, v, u) for (v, u) in zip(v, u))
    inds = GetInds(formula.Vars_) + GetInds(v.Vars_) + GetInds(u.Vars_)
    ind = 1 + max(inds) if len(inds) > 0 else 0
    dim = formula.dim
    cat = -1
    gradin = Var(ind, dim, cat)

    dFT = formula.DiffT(v, gradin)

    return AdjointOperator(dFT, gradin, u)
