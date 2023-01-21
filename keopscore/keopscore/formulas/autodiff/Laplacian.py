from keopscore.formulas import Var
from keopscore.formulas.LinearOperators import TraceOperator
from keopscore.utils.code_gen_utils import GetInds
from keopscore.formulas.variables.IntCst import IntCst
from keopscore.utils.misc_utils import KeOps_Error

# /////////////////////////////////////////////////////////////
# ///      LAPLACIAN OPERATOR  : Laplacian< F, V, U >       ////
# /////////////////////////////////////////////////////////////

# Defines Delta_v(F), the laplacian of F with respect to v
# F must be of dimension 1

def Laplacian(formula, v):
    
    if formula.dim != 1:
        KeOps_Error("Laplacian is only implemented for scalar formula.")
        
    dFT = formula.DiffT(v, IntCst(1))

    # we define a temporary gradin to backpropagate a second time.
    # This gradin will disappear in the formula when we compute the trace.
    # We define this temporary gradin as Var(ind,dim,cat)
    # with ind : a new index that must not exist in the formula,
    # dim : must equal dFT.dim = v.dim
    # cat : not specified and not used, we put -1
    inds = GetInds(formula.Vars_)
    ind = 1 + max(inds) if len(inds) > 0 else 0
    dim = v.dim
    cat = -1
    gradin = Var(ind, dim, cat)
    
    H = dFT.DiffT(v, gradin)
    
    return TraceOperator(H, gradin)