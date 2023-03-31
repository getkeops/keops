from keopscore.formulas.reductions.KMin_ArgKMin_Reduction import KMin_ArgKMin_Reduction
from keopscore.formulas.reductions.Zero_Reduction import Zero_Reduction
from keopscore.utils.code_gen_utils import (
    c_for_loop,
    new_c_varname,
    c_variable,
)


class ArgKMin_Reduction(KMin_ArgKMin_Reduction):
    """Implements the arg-k-min reduction operation : for each i or each j, find the indices of the
    k minimal values of Fij operation is vectorized: if Fij is vector-valued, arg-k-min is computed
    for each dimension."""

    string_id = "ArgKMin_Reduction"

    def __init__(self, formula, K, tagIJ):
        super().__init__(formula, K, tagIJ)
        self.dim = K * formula.dim

    def FinalizeOutput(self, acc, out, i):
        fdim = self.formula.dim
        p = c_variable("int", new_c_varname("p"))
        loop, k = c_for_loop(0, fdim, 1, pragma_unroll=True)
        body = p.declare_assign(k)
        inner_loop, l = c_for_loop(
            k, k + 2 * self.K * fdim, 2 * fdim, pragma_unroll=True
        )
        body += inner_loop(out[p].assign(acc[l + fdim]) + p.add_assign(fdim))
        return loop(body)
        outer_body

    def DiffT(self, v, gradin):
        return Zero_Reduction(v.dim, v.cat % 2)
