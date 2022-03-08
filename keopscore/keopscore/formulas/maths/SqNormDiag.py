from keopscore.formulas.maths.Square import Square
from keopscore.formulas.maths.Sum import Sum


class SqNormDiag:
    """
    Anisotropic (but diagonal) norm, if S.dim == A.dim: SqNormDiag(S,A) = sum_i s_i*a_i*a_i
    """

    def __new__(cls, S, A):
        return Sum(S * Square(A))

    enable_test = False
