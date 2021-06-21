from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.utils.code_gen_utils import c_variable, c_for_loop, c_zero_float

####################################
######    Tensor product       #####
####################################

class TensorProd(Operation):
    string_id = "TensorProd"

    def __init__(self, arg0, arg1):
        super().__init__(arg0, arg1)
        self.dim = arg0.dim * arg1.dim

    def Op(self, out, table, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        q = c_variable("int")
        loop, k = c_for_loop(0, arg0.dim, 1, pragma_unroll=True)
        inner_loop, l = c_for_loop(0, arg1.dim, 1, pragma_unroll=True)
        return f"""
                    #if C_CONTIGUOUS     // row major
                        {q.declare_assign(0)}
                        {loop( inner_loop( out[q].assign(arg0[k]*arg1[l]) + q.add_assign(1) ) )}
                    #else               // column major
                        {q.declare_assign(0)}
                        {loop( inner_loop( out[k + l * arg0.dim].assign(arg0[k]*arg1[l]) + q.add_assign(1) ) )}
                    #endif
                """

    def DiffT(self, v, gradin):
        from keops.python_engine.formulas import MatVecMult, VecMatMult
        f = self.children[0]
        g = self.children[1]
        return f.Grad(v, MatVecMult(gradin, g)) + g.Grad(v, VecMatMult(f, gradin))
