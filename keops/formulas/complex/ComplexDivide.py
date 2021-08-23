from keops.formulas.complex.Real2Complex import Real2Complex
from keops.formulas.complex.ComplexMult import ComplexMult
from keops.formulas.complex.ComplexSquareAbs import ComplexSquareAbs
from keops.formulas.complex.Conj import Conj
from keops.formulas.maths.Inv import Inv

# /////////////////////////////////////////////////////////////////////////
# ////      ComplexDivide                           ////
# /////////////////////////////////////////////////////////////////////////


def ComplexDivide(f, g):
    return ComplexMult(Real2Complex(Inv(ComplexSquareAbs(g))), ComplexMult(f, Conj(g)))
