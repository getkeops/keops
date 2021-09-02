from keops.formulas.maths import Sqrt
from keops.formulas.maths.Scalprod import Scalprod

class Norm2():
    def __new__(cls, arg):
        return Sqrt(Scalprod(arg, arg))

    enable_test = False
