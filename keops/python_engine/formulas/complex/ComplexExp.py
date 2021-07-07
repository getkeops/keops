from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.utils.code_gen_utils import (
    c_for_loop,
    new_c_varname,
    c_variable,
)
from keops.python_engine.utils.math_functions import keops_exp, keops_cos, keops_sin
from keops.python_engine.utils.code_gen_utils import c_variable
from keops.python_engine.formulas.complex.ComplexReal import ComplexReal
from keops.python_engine.formulas.complex.ComplexImag import ComplexImag
from keops.python_engine.formulas.complex.Real2Complex import Real2Complex
from keops.python_engine.formulas.complex.Imag2Complex import Imag2Complex
from keops.python_engine.formulas.maths.Exp import Exp
from keops.python_engine.formulas.maths.Cos import Cos
from keops.python_engine.formulas.maths.Sin import Sin


# /////////////////////////////////////////////////////////////////////////
# ////      ComplexExp                           ////
# /////////////////////////////////////////////////////////////////////////


class ComplexExp(Operation):

    string_id = "ComplexExp"

    def __init__(self, f):
        if f.dim % 2 != 0:
            raise ValueError("Dimension of F must be even")
        self.dim = f.dim
        super().__init__(f)

    def Op(self, out, table, inF):
        forloop, i = c_for_loop(0, out.dim, 2, pragma_unroll=True)
        r = c_variable(out.dtype, new_c_varname("r"))
        body = r.declare_assign(keops_exp(inF[i]))
        body += out[i].assign(r * keops_cos(inF[i + 1]))
        body += out[i + 1].assign(r * keops_sin(inF[i + 1]))
        return forloop(body)

    # building equivalent formula for autodiff
    def DiffT(self, v, gradin):
        f = self.children[0]
        AltAbs = Exp(ComplexReal(f))
        AltReal = AltAbs * Cos(ComplexImag(f))
        AltImag = AltAbs * Sin(ComplexImag(f))
        AltFormula = Real2Complex(AltReal) + Imag2Complex(AltImag)
        return AltFormula.DiffT(v, gradin)
