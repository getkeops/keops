from keopscore.formulas.Operation import Operation
from keopscore.utils.code_gen_utils import (
    c_variable,
    c_for_loop,
)
from keopscore.utils.misc_utils import KeOps_Error


####################################
######    Tensor product       #####
####################################


class TensorProd(Operation):
    string_id = "TensorProd"
    linearity_type = "one"

    def __init__(self, arg0, arg1, params=()):
        # N.B. params keyword is used for compatibility with base class, but should always equal ()
        if params != ():
            KeOps_Error("There should be no parameter.")
        super().__init__(arg0, arg1)
        self.dim = arg0.dim * arg1.dim

    def Op(self, out, table, arg0, arg1):
        q = c_variable("int")
        loop, k = c_for_loop(0, arg0.dim, 1, pragma_unroll=True)
        inner_loop, l = c_for_loop(0, arg1.dim, 1, pragma_unroll=True)
        return f"""
                    #if C_CONTIGUOUS     // row major
                        {q.declare_assign(0)}
                        {loop(inner_loop(out[q].assign(arg0[k] * arg1[l]) + q.add_assign(1)))}
                    #else               // column major
                        {q.declare_assign(0)}
                        {loop(inner_loop(out[k + l * arg0.dim].assign(arg0[k] * arg1[l]) + q.add_assign(1)))}
                    #endif
                """

    def DiffT(self, v, gradin):
        from keopscore.formulas import MatVecMult, VecMatMult

        f, g = self.children
        return f.DiffT(v, MatVecMult(gradin, g)) + g.DiffT(v, VecMatMult(f, gradin))
