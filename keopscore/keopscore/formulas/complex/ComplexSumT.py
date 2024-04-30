from keopscore.formulas.Operation import Operation
from keopscore.utils.meta_toolbox import c_for_loop
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.unique_object import unique_object

# /////////////////////////////////////////////////////////////////////////
# ////      adjoint of ComplexSum                           ////
# /////////////////////////////////////////////////////////////////////////


def ComplexSumT(f, dim):
    return ComplexSumT_Impl_Factory(dim)(f)


class ComplexSumT_Impl:
    pass


class ComplexSumT_Impl_Factory(metaclass=unique_object):

    def __init__(self, dim):

        class Class(ComplexSumT_Impl):

            string_id = "ComplexSumT"

            def __init__(self, f, dim):
                if f.dim != 2:
                    KeOps_Error("Dimension of F must be 2")
                self.dim = dim
                super().__init__(f)

            def Op(self, out, table, inF):
                forloop, i = c_for_loop(0, self.dim, 2, pragma_unroll=True)
                body = out[i].assign(inF[0])
                body += out[i + 1].assign(inF[1])
                return forloop(body)

            def DiffT(self, v, gradin):
                from keopscore.formulas.complex.ComplexSum import ComplexSum

                f = self.children[0]
                return f.DiffT(v, ComplexSum(gradin))

        self.Class = Class

    def __call__(self, f):
        return self.Class(f)
