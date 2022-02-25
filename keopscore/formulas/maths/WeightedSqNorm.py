from keopscore.formulas.maths.SqNormDiag import SqNormDiag
from keopscore.formulas.maths.SqNormIso import SqNormIso
from keopscore.formulas.maths.SymSqNorm import SymSqNorm


class WeightedSqNorm:
    """
    WeightedSqNorm(A,X) : redirects to SqNormIso, SqNormDiag or SymSqNorm depending on dimension of A.
    """

    string_id = "WeightedSqNorm"

    def __new__(cls, A, X):
        if A.dim == 1:
            return SqNormIso(A, X)
        elif A.dim == X.dim:
            return SqNormDiag(A, X)
        else:
            return SymSqNorm(A, X)

    enable_test = True
    nargs = 2  # number of arguments
    test_argdims = [5, 5]  # dimensions of arguments for testing
