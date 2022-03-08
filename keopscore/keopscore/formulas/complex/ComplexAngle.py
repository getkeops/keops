from keopscore.formulas.maths.Atan2 import Atan2
from keopscore.formulas.complex.ComplexReal import ComplexReal
from keopscore.formulas.complex.ComplexImag import ComplexImag

# /////////////////////////////////////////////////////////////////////////
# ////      ComplexAngle                           ////
# /////////////////////////////////////////////////////////////////////////


def ComplexAngle(f):
    return Atan2(ComplexImag(f), ComplexReal(f))
