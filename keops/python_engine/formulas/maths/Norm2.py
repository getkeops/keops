from keops.python_engine.formulas.maths import Sqrt, Scalprod

##########################
######    Norm2      #####
##########################

def Norm2(arg):
    return Sqrt(Scalprod(arg, arg))