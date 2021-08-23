from keops.formulas.maths.Atan2 import Atan2
from keops.formulas.complex.ComplexReal import ComplexReal
from keops.formulas.complex.ComplexImag import ComplexImag

# /////////////////////////////////////////////////////////////////////////
# ////      ComplexAngle                           ////
# /////////////////////////////////////////////////////////////////////////


def ComplexAngle(f):
    return Atan2(ComplexImag(f), ComplexReal(f))
