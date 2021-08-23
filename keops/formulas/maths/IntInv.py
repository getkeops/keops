from keops.formulas.maths.Inv import Inv
from keops.formulas.variables.IntCst import IntCst


##########################
######    IntInv     #####
##########################


def IntInv(arg):
    return Inv(IntCst(arg))
