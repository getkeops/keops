from keopscore.formulas.complex.Real2Complex import Real2Complex
from keopscore.formulas.complex.ComplexMult import ComplexMult
from keopscore.formulas.complex.ComplexSquareAbs import ComplexSquareAbs
from keopscore.formulas.complex.Conj import Conj
from keopscore.formulas.maths.Inv import Inv

# /////////////////////////////////////////////////////////////////////////
# ////      ComplexDivide                           ////
# /////////////////////////////////////////////////////////////////////////


def ComplexDivide(f, g):
    return ComplexMult(Real2Complex(Inv(ComplexSquareAbs(g))), ComplexMult(f, Conj(g)))
