from keops.formulas.maths.SqNorm2 import SqNorm2


###########################################################################
######   WEIGHTED SQUARED DISTANCE : WeightedSqDist(S,A)    ###############
###########################################################################


def WeightedSqDist(S, A):
    return S * SqNorm2(A)
