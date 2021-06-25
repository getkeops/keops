from keops.python_engine.formulas.vectOps.Sum import Sum
from keops.python_engine.formulas.vectOps.TensorProd import TensorProd

###########################################################################
####       Fully anisotropic norm, if S.dim == A.dim * A.dim          #####
###########################################################################

# SymSqNorm(A,X) = sum_{ij} a_ij * x_i*x_j


def SymSqNorm(A, X):
    return Sum(A * TensorProd(X, X))
