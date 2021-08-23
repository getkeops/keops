from keops.formulas.maths.SqNorm2 import SqNorm2


###############################################################
######     ISOTROPIC NORM : SqNormIso(S,A)    #################
###############################################################


def SqNormIso(S, A):
    return S * SqNorm2(A)
