from keopscore.formulas.maths.ElemT import ElemT
from keopscore.formulas.maths.Concat import Concat
from keopscore.formulas.variables.IntCst import IntCst
from keopscore.formulas.variables.Var import Var
from keopscore.utils.code_gen_utils import GetInds


# //////////////////////////////////////////////////////////////////////////////////////////////
# ////      Standard basis of R^DIM : < (1,0,0,...) , (0,1,0,...) , ... , (0,...,0,1) >     ////
# //////////////////////////////////////////////////////////////////////////////////////////////


def StandardBasis(dim):
    return tuple(ElemT(IntCst(1), dim, i) for i in range(dim))


# /////////////////////////////////////////////////////////////////////////
# ////      Matrix of gradient operator (=transpose of jacobian)       ////
# /////////////////////////////////////////////////////////////////////////


class GradMatrix:
    def __new__(cls, f, v):
        f.Vars(cat=3)
        IndsTempVars = GetInds(f.Vars(cat=3))
        newind = 1 if len(IndsTempVars) == 0 else 1 + max(IndsTempVars)
        gradin = Var(newind, f.dim, 3)
        packGrads = tuple(
            f.DiffT(v, gradin).replace(gradin, e) for e in StandardBasis(f.dim)
        )
        res = packGrads[0]
        for elem in packGrads[1:]:
            res = Concat(res, elem)
        return res

    enable_test = False
