from keops.python_engine.formulas.vectOps.SqNormIso import SqNormIso
from keops.python_engine.formulas.vectOps.SqNormDiag import SqNormDiag
from keops.python_engine.formulas.vectOps.SymSqNorm import SymSqNorm

# WeightedSqNorm(A,X) : redirects to SqNormIso, SqNormDiag or SymSqNorm
# depending on dimension of A.


def WeightedSqNorm(A, X):
    if A.dim == 1:
        return SqNormIso(A, X)
    elif A.dim == X.dim:
        return SqNormDiag(A, X)
    else:
        return SymSqNorm(A, X)