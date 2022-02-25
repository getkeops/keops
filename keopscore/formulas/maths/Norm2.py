from keopscore.formulas.maths import Sqrt
from keopscore.formulas.maths.Scalprod import Scalprod


class Norm2:
    def __new__(cls, arg):
        return Sqrt(Scalprod(arg, arg))

    enable_test = False
