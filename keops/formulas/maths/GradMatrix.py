from keops.formulas.maths.ElemT import ElemT
from keops.formulas.maths.Concat import Concat
from keops.formulas.variables.IntCst import IntCst
from keops.formulas.variables.Var import Var
from keops.utils.code_gen_utils import GetInds


# //////////////////////////////////////////////////////////////////////////////////////////////
# ////      Standard basis of R^DIM : < (1,0,0,...) , (0,1,0,...) , ... , (0,...,0,1) >     ////
# //////////////////////////////////////////////////////////////////////////////////////////////

def StandardBasis(dim):
    return tuple(ElemT(IntCst(1), dim, i) for i in range(dim))


# /////////////////////////////////////////////////////////////////////////
# ////      Matrix of gradient operator (=transpose of jacobian)       ////
# /////////////////////////////////////////////////////////////////////////


def GradMatrix(f, v):
    f.Vars(cat=3)
    IndsTempVars = GetInds(f.Vars(cat=3))
    newind = 1 if len(IndsTempVars)==0 else 1+max(IndsTempVars)
    gradin = Var(newind, f.dim, 3)
    packGrads = tuple(f.DiffT(v, gradin).replace(gradin, e) for e in StandardBasis(f.dim))
    res = packGrads[0]
    for elem in packGrads[1:]:
        res = Concat(res, elem)
    return res

