from keops.python_engine.formulas.maths import Square
from keops.python_engine.formulas.vectOps import Sum

#############################
######    SqNormDiag    #####
#############################

# Anisotropic (but diagonal) norm, if S.dim == A.dim:
# SqNormDiag(S,A) = sum_i s_i*a_i*a_i


def SqNormDiag(S, A):
    return Sum(S * Square(A))
