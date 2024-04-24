from keopscore.formulas.Operation import Operation
from keopscore.utils.meta_toolbox import c_for_loop
from keopscore.utils.misc_utils import KeOps_Error

# /////////////////////////////////////////////////////////////////////////
# ////      adjoint of ComplexSum                           ////
# /////////////////////////////////////////////////////////////////////////


class ComplexSumT(Operation):
    string_id = "ComplexSumT"

    def __init__(self, f, dim=None, params=None):
        # N.B. init via params keyword is used for compatibility with base class.
        if dim is None:
            # here params should be a tuple containing one single integer
            (dim,) = params
        if f.dim != 2:
            KeOps_Error("Dimension of F must be 2")
        self.dim = dim
        super().__init__(f, params=(dim,))

    def Op(self, out, table, inF):
        forloop, i = c_for_loop(0, self.dim, 2, pragma_unroll=True)
        body = out[i].assign(inF[0])
        body += out[i + 1].assign(inF[1])
        return forloop(body)

    def DiffT(self, v, gradin):
        from keopscore.formulas.complex.ComplexSum import ComplexSum

        f = self.children[0]
        return f.DiffT(v, ComplexSum(gradin))
