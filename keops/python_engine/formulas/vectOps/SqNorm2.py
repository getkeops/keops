from keops.python_engine.formulas.vectOps.Scalprod import Scalprod

##########################
######    SqNorm2    #####
##########################


def SqNorm2(arg0):
    return Scalprod(arg0, arg0)