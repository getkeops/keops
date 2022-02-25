from keopscore.formulas.maths.Sum import Sum
from keopscore.formulas.maths.TensorProd import TensorProd


class SymSqNorm:
    """
    Fully anisotropic norm, if S.dim == A.dim * A.dim
    SymSqNorm(A,X) = sum_{ij} a_ij * x_i*x_j
    """

    def __new__(cls, A, X):
        return Sum(A * TensorProd(X, X))

    enable_test = False
