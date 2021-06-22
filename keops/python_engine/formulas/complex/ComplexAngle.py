
from keops.python_engine.formulas.maths.Atan2 import Atan2
from keops.python_engine.formulas.complex.ComplexReal import ComplexReal
from keops.python_engine.formulas.complex.ComplexImag import ComplexImag

#/////////////////////////////////////////////////////////////////////////
#////      ComplexAngle                           ////
#/////////////////////////////////////////////////////////////////////////

def ComplexAngle(f):
    return Atan2(ComplexImag(f), ComplexReal(f))
