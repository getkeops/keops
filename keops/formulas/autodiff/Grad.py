from keops.formulas import Var
from keops.utils.code_gen_utils import GetInds

# /////////////////////////////////////////////////////////////
# ///      GRADIENT OPERATOR  : Grad< F, V, Gradin >       ////
# /////////////////////////////////////////////////////////////

# Defines [\partial_V F].gradin function
# Symbolic differentiation is a straightforward recursive operation,
# provided that the operators have implemented their DiffT "compiler methods":


def Grad(formula, v, gradin=None):
    if gradin is None:
        if v.cat == 2:
            raise ValueError("not implemented")
        inds = GetInds(formula.Vars_)
        ind = 1 + max(inds) if len(inds) > 0 else 0
        dim = formula.dim
        cat = 1 - v.cat
        gradin = Var(ind, dim, cat)
    return formula.DiffT(v, gradin)