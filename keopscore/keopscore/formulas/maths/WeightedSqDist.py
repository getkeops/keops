from keopscore.formulas.maths.WeightedSqNorm import WeightedSqNorm


class WeightedSqDist:
    """
    WEIGHTED SQUARED DISTANCE : WeightedSqDist(S,F,G)
    """

    def __new__(cls, S, F, G):
        return WeightedSqNorm(S, F - G)

    enable_test = False
