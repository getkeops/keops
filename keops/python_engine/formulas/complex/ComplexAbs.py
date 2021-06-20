from keops.python_engine.formulas.complex.ComplexSquareAbs import ComplexSquareAbs
from keops.python_engine.formulas.maths.Sqrt import Sqrt


#/////////////////////////////////////////////////////////////////////////
#////      ComplexAbs                           ////
#/////////////////////////////////////////////////////////////////////////

def ComplexAbs(f):
    return Sqrt(ComplexSquareAbs(f));
