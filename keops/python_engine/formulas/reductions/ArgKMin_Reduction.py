from keops.python_engine.formulas.reductions.KMin_ArgKMin_Reduction import KMin_ArgKMin_Reduction
from keops.python_engine.formulas.reductions.Zero_Reduction import Zero_Reduction


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
        return f"""
                        #pragma unroll
                        for(int k=0; k<{fdim}; k++)
                            #pragma unroll
                            for(int p=k, l=k; l<{self.K}*2*{fdim}+k; p+={fdim}, l+=2*{fdim})
                                {out.id}[p] = {acc.id}[l+{fdim}];
                """

    def DiffT(self, v, gradin):
        return Zero_Reduction(v.dim, v.cat % 2)