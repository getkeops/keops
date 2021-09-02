from keops.formulas.maths.Inv import Inv
from keops.formulas.variables.IntCst import IntCst


class IntInv():
    def __new__(cls, arg):
        return Inv(IntCst(arg))

    enable_test = False
