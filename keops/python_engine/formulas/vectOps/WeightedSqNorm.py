from keops.python_engine.formulas.vectOps import SqNormIso, SqNormDiag, SymSqNorm

# WeightedSqNorm(A,X) : redirects to SqNormIso, SqNormDiag or SymSqNorm
# depending on dimension of A.


def WeightedSqNorm(A, X):
    if A.dim == 1:
        return SqNormIso(A, X)
    elif A.dim == X.dim:
        return SqNormDiag(A, X)
    else:
        return SymSqNorm(A, X)