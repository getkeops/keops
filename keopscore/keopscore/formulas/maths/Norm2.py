from keopscore.formulas.maths.Sqrt import Sqrt
from keopscore.formulas.maths.Scalprod import Scalprod


class Norm2:
    def __new__(cls, arg):
        return Sqrt(Scalprod(arg, arg))

    enable_test = False
