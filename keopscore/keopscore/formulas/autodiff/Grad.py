from keopscore.formulas import Var, Concat
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
        # TODO : check why we do this here.
        # Is this case gradin=None used anywhere in the code ??
        # Why gradin.cat should be 1-v.cat ?
        # Why v.cat should not be 2 ?
        if v.cat == 2:
            KeOps_Error("not implemented")
        inds = GetInds(formula.Vars_)
        ind = 1 + max(inds) if len(inds) > 0 else 0
        dim = formula.dim
        cat = 1 - v.cat
        gradin = Var(ind, dim, cat)
    if isinstance(v, list):
        return Concat(*(formula.DiffT(u, gradin) for u in v))
    else:
        return formula.DiffT(v, gradin)
