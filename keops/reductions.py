from keops.formulas import *
from keops.utils.code_gen_utils import get_hash_name, GetInds


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


# same with additional saved forward variable. This is only used for taking gradients of reductions operations.
def Grad_WithSavedForward(red_formula, v, gradin, f0):
    return red_formula.DiffT(v, gradin, f0)


class getReduction:
    library = {}

    def __new__(self, red_formula_string, aliases=[]):
        aliases_dict = {}
        for alias in aliases:
            if "=" in alias:
                varname, var = alias.split("=")
                aliases_dict[varname] = eval(var)
        string_id_hash = get_hash_name(red_formula_string, aliases)
        if string_id_hash in getReduction.library:
            return getReduction.library[string_id_hash]
        else:
            reduction = eval(red_formula_string, globals(), aliases_dict)
            getReduction.library[string_id_hash] = reduction
            return reduction
