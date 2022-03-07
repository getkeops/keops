from keopscore.formulas.maths.Inv import Inv
from keopscore.formulas.variables.IntCst import IntCst


class IntInv:
    def __new__(cls, arg):
        return Inv(IntCst(arg))

    enable_test = False
