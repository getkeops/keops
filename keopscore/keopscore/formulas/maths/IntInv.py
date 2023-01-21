from keopscore.formulas.maths.Inv import Inv
from keopscore.formulas.variables.RatCst import RatCst


class IntInv:
    def __new__(cls, arg):
        return RatCst(1,arg)

    enable_test = False
