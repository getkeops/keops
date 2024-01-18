from keopscore.formulas.maths.Sum import Sum


class SqNorm2:
    def __new__(cls, arg0):
        return Sum(arg0**2)

    enable_test = False
