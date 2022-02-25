from keopscore.formulas.complex.ComplexSquareAbs import ComplexSquareAbs
from keopscore.formulas.maths.Sqrt import Sqrt


# /////////////////////////////////////////////////////////////////////////
# ////      ComplexAbs                           ////
# /////////////////////////////////////////////////////////////////////////


def ComplexAbs(f):
    return Sqrt(ComplexSquareAbs(f))
