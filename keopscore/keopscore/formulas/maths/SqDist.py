from keopscore.formulas.maths.SqNorm2 import SqNorm2


class SqDist:
    def __new__(cls, arg0, arg1):
        return SqNorm2(arg0 - arg1)

    enable_test = False
