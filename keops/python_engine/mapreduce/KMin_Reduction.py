from keops.python_engine.mapreduce.KMin_ArgKMin_Reduction import KMin_ArgKMin_Reduction


class KMin_Reduction(KMin_ArgKMin_Reduction):
    """Implements the k-min reduction operation : for each i or each j, find the
     k minimal values of Fij operation is vectorized: if Fij is vector-valued, arg-k-min
     is computed for each dimension."""

    string_id = "KMin_Reduction"

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
                                {out.id}[p] = {acc.id}[l];
                """
