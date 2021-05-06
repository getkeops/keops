from keops.python_engine.code_gen_utils import infinity, cast_to
from keops.python_engine.mapreduce.Reduction import Reduction


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
            raise ValueError("not implemented")
        return f"""
                    #pragma unroll
                    for(int k=0; k<{self.formula.dim}; k++) {{
                        #pragma unroll
                        for(int l=k; l<{self.K}*2*{self.formula.dim}+k; l+=2*{self.formula.dim}) {{
                            {acc.id}[l] = {infinity(acc.dtype)}; // initialize output
                            {acc.id}[l+{self.formula.dim}] = {cast_to(acc.dtype)} 0.0f; // initialize output
                        }}
                    }}
                """

    def ReducePair(self, acc, xi):
        # Returns C++ code that implements the update phase of the reduction.
        dtype = xi.dtype
        fdim = self.formula.dim
        return f"""
                    {{
                        {dtype} out[{self.dimred}];
                        #pragma unroll
                        for(int k=0; k<{fdim}; k++) {{
                            int p = k;
                            int q = k;
                            #pragma unroll
                            for(int l=k; l<{self.dimred}; l+=2*{fdim}) {{
                                if({xi.id}[p]<{acc.id}[q]) {{
                                    out[l] = {xi.id}[p];
                                    out[{fdim}+l] = {xi.id}[{fdim}+p];
                                    p += 2*{fdim};
                                }}
                                else {{
                                    out[l] = {acc.id}[q];
                                    out[{fdim}+l] = {acc.id}[{fdim}+q];
                                    q += 2*{fdim};
                                }}  
                            }}
                        }}
                        #pragma unroll
                        for(int k=0; k<{self.dimred}; k++)
                            {acc.id}[k] = out[k];
                    }}
                """

    def ReducePairShort(self, acc, xi, ind):
        fdim = self.formula.dim
        dtype = xi.dtype
        return f"""
                    {{
                        {dtype} xik;
                        int l;
                        #pragma unroll
                        for(int k=0; k<{fdim}; k++) {{
                            xik = {xi.id}[k];
                            #pragma unroll
                            for(l=({self.K}-1)*2*{fdim}+k; l>=k && xik<{acc.id}[l]; l-=2*{fdim}) {{
                                {dtype} tmpl = {acc.id}[l];
                                int indtmpl = {acc.id}[l+{fdim}];
                                {acc.id}[l] = xik;
                                {acc.id}[l+{fdim}] = {ind.id};
                                if(l<({self.K}-1)*2*{fdim}+k) {{
                                    {acc.id}[l+2*{fdim}] = tmpl;
                                    {acc.id}[l+2*{fdim}+{fdim}] = indtmpl;
                                }}
                            }}
                        }}
                    }}
                """
