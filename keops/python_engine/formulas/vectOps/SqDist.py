from keops.python_engine.formulas.vectOps.SqNorm2 import SqNorm2

##########################
######    SqDist     #####
##########################


def SqDist(arg0, arg1):
    return SqNorm2(arg0 - arg1)
