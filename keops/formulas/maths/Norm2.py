from keops.formulas.maths import Sqrt
from keops.formulas.maths.Scalprod import Scalprod


##########################
######    Norm2      #####
##########################


def Norm2(arg):
    return Sqrt(Scalprod(arg, arg))
