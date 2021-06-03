from keops.python_engine.formulas.maths import Rsqrt, SqNorm2


##########################
####    Normalize    #####
##########################


def Normalize(arg):
    return Rsqrt(SqNorm2(arg)) * arg
