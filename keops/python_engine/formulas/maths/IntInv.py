from keops.python_engine.formulas.maths.Inv import Inv
from keops.python_engine.formulas.variables.IntCst import IntCst


##########################
######    IntInv     #####
##########################


def IntInv(arg):
    return Inv(IntCst(arg))
