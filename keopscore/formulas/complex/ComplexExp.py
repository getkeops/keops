from keopscore.formulas.VectorizedComplexScalarOp import VectorizedComplexScalarOp
from keopscore.utils.code_gen_utils import (
    c_for_loop,
    new_c_varname,
    c_variable,
)
from keopscore.utils.math_functions import keops_exp, keops_cos, keops_sin
from keopscore.utils.code_gen_utils import c_variable
from keopscore.formulas.complex.ComplexReal import ComplexReal
from keopscore.formulas.complex.ComplexImag import ComplexImag
from keopscore.formulas.complex.Real2Complex import Real2Complex
from keopscore.formulas.complex.Imag2Complex import Imag2Complex
from keopscore.formulas.maths.Exp import Exp
from keopscore.formulas.maths.Cos import Cos
from keopscore.formulas.maths.Sin import Sin


# /////////////////////////////////////////////////////////////////////////
# ////      ComplexExp                           ////
# /////////////////////////////////////////////////////////////////////////


class ComplexExp(VectorizedComplexScalarOp):

    string_id = "ComplexExp"

    def ScalarOp(self, out, inF):
        r = c_variable(out.dtype, new_c_varname("r"))
        string = r.declare_assign(keops_exp(inF[0]))
        string += out[0].assign(r * keops_cos(inF[1]))
        string += out[1].assign(r * keops_sin(inF[1]))
        return string

    # building equivalent formula for autodiff
    def DiffT(self, v, gradin):
        f = self.children[0]
        AltAbs = Exp(ComplexReal(f))
        AltReal = AltAbs * Cos(ComplexImag(f))
        AltImag = AltAbs * Sin(ComplexImag(f))
        AltFormula = Real2Complex(AltReal) + Imag2Complex(AltImag)
        return AltFormula.DiffT(v, gradin)
