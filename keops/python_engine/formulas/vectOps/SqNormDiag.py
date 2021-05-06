from keops.python_engine.formulas.basicMathOps import Sum, Square

#############################
######    SqNormDiag    #####
#############################

# Anisotropic (but diagonal) norm, if S.dim == A.dim:
# SqNormDiag(S,A) = sum_i s_i*a_i*a_i


def SqNormDiag(S, A):
    return Sum(S * Square(A))
