from keopscore.utils.code_gen_utils import (
    infinity,
    cast_to,
    c_zero_float,
    c_for_loop,
    c_variable,
    new_c_varname,
    c_if,
    c_array,
    use_pragma_unroll,
)
from keopscore.formulas.reductions.Reduction import Reduction
from keopscore.utils.misc_utils import KeOps_Error


class KMin_ArgKMin_Reduction(Reduction):
    """Implements the k-min-arg-k-min reduction operation : for each i or each j, find the
    values and indices of the k minimal values of Fij operation is vectorized: if Fij is vector-valued,
     arg-k-min is computed for each dimension."""

    string_id = "KMin_ArgKMin_Reduction"

    def __init__(self, formula, K, tagIJ):
        super().__init__(formula, tagIJ)

        self.K = K

        # dim is dimension of output of reduction ; for a arg-k-min reduction it is equal to the dimension of output of formula
        self.dim = 2 * K * formula.dim

        # We work with a (values,indices) vector
        self.dimred = self.dim  # dimension of inner reduction variables

    def InitializeReduction(self, acc):
        # Returns C++ code to be used at initialization phase of the reduction.
        if acc.dtype == "half2":
            KeOps_Error("not implemented")
        fdim, K = self.formula.dim, self.K
        outer_loop, k = c_for_loop(0, fdim, 1, pragma_unroll=True)
        inner_loop, l = c_for_loop(k, k + (2 * K * fdim), 2 * fdim, pragma_unroll=True)
        return outer_loop(
            inner_loop(
                acc[l].assign(infinity(acc.dtype)) + acc[l + fdim].assign(c_zero_float)
            )
        )

    def ReducePair(self, acc, xi):
        # Returns C++ code that implements the update phase of the reduction.
        dtype = xi.dtype
        fdim = self.formula.dim
        out = c_array(dtype, self.dimred, new_c_varname("out"))
        outer_loop, k = c_for_loop(0, fdim, 1)
        p = c_variable("int", new_c_varname("p"))
        q = c_variable("int", new_c_varname("q"))
        inner_loop, l = c_for_loop(k, self.dimred, 2 * fdim)
        inner_body = c_if(
            xi[p] < acc[q],
            out[l].assign(xi[p])
            + out[l + fdim].assign(xi[p + fdim])
            + p.add_assign(2 * fdim),
            out[l].assign(acc[q])
            + out[l + fdim].assign(acc[q + fdim])
            + q.add_assign(2 * fdim),
        )
        outer_body = p.declare_assign(k) + q.declare_assign(k) + inner_loop(inner_body)
        final_loop, k = c_for_loop(0, self.dimred, 1)
        return (
            out.declare() + outer_loop(outer_body) + final_loop(acc[k].assign(out[k]))
        )

    def ReducePairShort(self, acc, xi, ind):
        fdim, K = self.formula.dim, self.K
        dtype = xi.dtype

        xik = c_variable(dtype, new_c_varname("xik"))
        l = c_variable("int", new_c_varname("l"))
        k = c_variable("int", new_c_varname("k"))
        tmpl = c_variable(dtype, new_c_varname("tmpl"))
        indtmpl = c_variable("int", new_c_varname("indtmpl"))
        return f"""
                    {{
                        {xik.declare()}
                        {l.declare()}
                        {use_pragma_unroll()}
                        for(int {k.id}=0; {k.id}<{fdim}; {k.id}++) {{
                            {xik.assign(xi[k])}
                            {use_pragma_unroll()}                 
                            for({l.id}={(k+(K-1)*2*fdim).id}; {l.id}>={k.id} && {(xik<acc[l]).id}; {l.id}-={2*fdim}) {{
                                {tmpl.declare_assign(acc[l])}
                                {indtmpl.declare_assign(acc[l+fdim])}
                                {acc[l].assign(xik)}
                                {acc[l+fdim].assign(ind)}                      
                                if({l.id}<{(k+(2*fdim*(K-1))).id}) {{
                                    {acc[l+2*fdim].assign(tmpl)}
                                    {acc[l+2*fdim+fdim].assign(indtmpl)}
                                }}
                            }}
                        }}
                    }}
                """
