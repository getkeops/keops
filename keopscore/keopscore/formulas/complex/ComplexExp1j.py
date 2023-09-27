from keopscore.formulas.Operation import Operation
from keopscore.utils.code_gen_utils import c_for_loop, pointer
from keopscore.utils.math_functions import keops_sincos
from keopscore.formulas.complex.Real2Complex import Real2Complex
from keopscore.formulas.complex.Imag2Complex import Imag2Complex
from keopscore.formulas.maths.Cos import Cos
from keopscore.formulas.maths.Sin import Sin
from keopscore.utils.misc_utils import KeOps_Error

# /////////////////////////////////////////////////////////////////////////
# ////      ComplexExp1j                           ////
# /////////////////////////////////////////////////////////////////////////


class ComplexExp1j(Operation):
    string_id = "ComplexExp1j"

    def __init__(self, f, params=()):
        # N.B. params keyword is used for compatibility with base class, but should always equal ()
        if params != ():
            KeOps_Error("There should be no parameter.")
        self.dim = 2 * f.dim
        super().__init__(f)

    def Op(self, out, table, inF):
        forloop, i = c_for_loop(0, self.dim, 2, pragma_unroll=True)
        body = keops_sincos(inF[i / 2], pointer(out[i + 1]), pointer(out[i]))
        return forloop(body)

    def DiffT(self, v, gradin):
        # building equivalent formula for autodiff
        f = self.children[0]
        AltFormula = Real2Complex(Cos(f)) + Imag2Complex(Sin(f))
        return AltFormula.DiffT(v, gradin)
