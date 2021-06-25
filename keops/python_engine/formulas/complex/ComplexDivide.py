from keops.python_engine.formulas.complex.Real2Complex import Real2Complex
from keops.python_engine.formulas.complex.ComplexMult import ComplexMult
from keops.python_engine.formulas.complex.ComplexSquareAbs import ComplexSquareAbs
from keops.python_engine.formulas.complex.Conj import Conj
from keops.python_engine.formulas.basicMathOps.Inv import Inv

# /////////////////////////////////////////////////////////////////////////
# ////      ComplexDivide                           ////
# /////////////////////////////////////////////////////////////////////////


def ComplexDivide(f, g):
    return ComplexMult(Real2Complex(Inv(ComplexSquareAbs(g))), ComplexMult(f, Conj(g)))
