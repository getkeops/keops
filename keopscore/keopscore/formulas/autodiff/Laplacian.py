from keopscore.formulas import Var
from keopscore.formulas.LinearOperators import TraceOperator
from keopscore.formulas.autodiff.Divergence import Divergence
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

    return Divergence(dFT, v)
