from keopscore.formulas.reductions.KMin_ArgKMin_Reduction import KMin_ArgKMin_Reduction
from keopscore.utils.code_gen_utils import (
    infinity,
    cast_to,
    c_zero_float,
    c_for_loop,
    c_variable,
    new_c_varname,
    c_if,
)


class KMin_Reduction(KMin_ArgKMin_Reduction):
    """Implements the k-min reduction operation : for each i or each j, find the
    k minimal values of Fij operation is vectorized: if Fij is vector-valued, arg-k-min
    is computed for each dimension."""

    string_id = "KMin_Reduction"

    def __init__(self, formula, K, tagIJ):
        super().__init__(formula, K, tagIJ)
        self.dim = K * formula.dim

    def FinalizeOutput(self, acc, out, i):
        fdim, K = self.formula.dim, self.K
        outer_loop, k = c_for_loop(0, fdim, 1)
        inner_loop, l = c_for_loop(k, k + (2 * fdim * K), 2 * fdim)
        p = c_variable("int", new_c_varname("p"))
        return outer_loop(
            p.declare_assign(k) + inner_loop(out[p].assign(acc[l]) + p.add_assign(fdim))
        )
