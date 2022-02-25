from keopscore.formulas.maths.SqNorm2 import SqNorm2


class SqNormIso:
    """
    ISOTROPIC NORM : SqNormIso(S,A)
    """

    def __new__(cls, S, A):
        return S * SqNorm2(A)

    enable_test = False
