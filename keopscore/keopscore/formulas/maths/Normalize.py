from keopscore.formulas.maths.Rsqrt import Rsqrt
from keopscore.formulas.maths.SqNorm2 import SqNorm2


class Normalize:
    def __new__(cls, arg):
        return Rsqrt(SqNorm2(arg)) * arg

    enable_test = False
