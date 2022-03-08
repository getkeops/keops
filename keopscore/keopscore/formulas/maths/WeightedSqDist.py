from keopscore.formulas.maths.SqNorm2 import SqNorm2


class WeightedSqDist:
    """
    WEIGHTED SQUARED DISTANCE : WeightedSqDist(S,A)
    """

    def __new__(cls, S, A):
        return S * SqNorm2(A)

    enable_test = False
