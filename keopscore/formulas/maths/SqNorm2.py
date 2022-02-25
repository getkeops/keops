from keopscore.formulas.maths.Scalprod import Scalprod


class SqNorm2:
    def __new__(cls, arg0):
        return Scalprod(arg0, arg0)

    enable_test = False
