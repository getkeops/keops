from keops.python_engine.formulas.maths import SqNorm2


###########################################################################
######   WEIGHTED SQUARED DISTANCE : WeightedSqDist(S,A)    ###############
###########################################################################


def WeightedSqDist(S, A):
    return S * SqNorm2(A)