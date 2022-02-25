from keopscore.formulas import Var
from keopscore.utils.code_gen_utils import GetInds
from keopscore.utils.misc_utils import KeOps_Error

# /////////////////////////////////////////////////////////////
# ///      GRADIENT OPERATOR  : Grad< F, V, Gradin >       ////
# /////////////////////////////////////////////////////////////

# Defines [\partial_V F].gradin function
# Symbolic differentiation is a straightforward recursive operation,
# provided that the operators have implemented their DiffT "compiler methods":


def Grad(formula, v, gradin=None):
    if gradin is None:
        if v.cat == 2:
            KeOps_Error("not implemented")
        inds = GetInds(formula.Vars_)
        ind = 1 + max(inds) if len(inds) > 0 else 0
        dim = formula.dim
        cat = 1 - v.cat
        gradin = Var(ind, dim, cat)
    return formula.DiffT(v, gradin)
