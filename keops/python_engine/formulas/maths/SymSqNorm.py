from keops.python_engine.formulas.maths import Sum, TensorProd

###########################################################################
####       Fully anisotropic norm, if S.dim == A.dim * A.dim          #####
###########################################################################

# SymSqNorm(A,X) = sum_{ij} a_ij * x_i*x_j

def SymSqNorm(A, X):
    return Sum(A * TensorProd(X, X))
