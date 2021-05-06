from keops.python_engine.formulas.vectOps import SqNorm2


###########################################################################
######   WEIGHTED SQUARED DISTANCE : WeightedSqDist(S,A)    ###############
###########################################################################


def WeightedSqDist(S, A):
    return S * SqNorm2(A)