from keops.python_engine.formulas.maths.Rsqrt import Rsqrt
from keops.python_engine.formulas.maths.SqNorm2 import SqNorm2


##########################
####    Normalize    #####
##########################


def Normalize(arg):
    return Rsqrt(SqNorm2(arg)) * arg
