from tree_class import tree
from operations import *
from utils import *    




class Reduction(tree):
    
    # Base class for all KeOps final reductions over a formula
    
    def __init__(self, formula, tagI):
        # - formula is an object of type Operation, it is the formula on which we apply a reduction
        # - tagI : 0 or 1, specifies wether we do the reduction over "i"-indexed or "j"-indexed variables.
        
        # We initialize several constants, most of them infered from the formula
        self.formula = formula
        self.children = [formula]
        self.params = (tagI,)
        self.tagI = tagI
        self.tagJ = 1-tagI
        self.cat = tagI
        self.Vars_ = formula.Vars_
        
    def ReducePair(self, acc, xi):
        # Returns C++ code that implements the update phase of the reduction.
        # by default it consists in a vectorized version of the ReducePairScalar operation.        
        return VectApply(self.ReducePairScalar, acc, xi)
        
    def ReducePairShort(self, acc, xi, ind):
        # N.B next lines are useless here, but to be used in other reductions :
        # if xi.dtype == "half2":
        #     half2_val = c_variable("half2_ind")
        #     string = half2_val.declare_assign(f"__floats2half2_rn(2*{ind()},2*{ind()}+1)")
        return self.ReducePair(acc, xi)

    def FinalizeOutput(self, acc, out, i):
        # Returns C++ code that implements the final output of the reduction.
        # For most reducitons it is a simple copy of the temporary variable
        # updated during the reduction, with possibly a cast if the accumulator was of
        # different data type.
        return VectCopy(out, acc)
    
class Zero_Reduction(Reduction):
    # Implements the zero reduction operation (fills output with zeros).
    # N.B. The actual code for filling zeros is not here ; when a Zero_reduction is detected,
    # the map_reduce scheme is redirected to CpuAssignZero or GpuAssignZero

    string_id = "Zero_Reduction"
    
    def __init__(self, dim, tagIJ):
        super().__init__(Zero(dim), tagIJ)
        self.dim = dim
        
    def DiffT(self, v, gradin, f0=None):
        return Zero_Reduction(v.dim,v.cat%2)
    
class Sum_Reduction(Reduction):
    # Sum reduction class
    string_id = "Sum_Reduction"
    
    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = formula.dim                      # dimension of final output of reduction
        self.dimred = self.dim                      # dimension of inner reduction variables
        self.dim_kahan = self.dim
        
    def InitializeReduction(self, tmp):
        # Returns C++ code to be used at initialization phase of the reduction.
        # Here it consists in setting the output array to zero.
        return tmp.assign(c_zero_float)
        
    def ReducePairScalar(self, tmp, xi):
        # Subroutine of ReducePairShort and ReducePair methods.
        # Returns C++ code that implements the "+=" accumulation operation of the sum reduction
        return f"{tmp.id} += {cast_to(tmp.dtype)}({xi.id});"
        
    def KahanScheme(self, acc, xi, tmp):
        return f"""
                        #pragma unroll
                        for (int k=0; k<{self.dim}; k++) 
                        {{
                            {acc.dtype} a = ({acc.dtype}) ({xi.id}[k] - {tmp.id}[k]);
                            {acc.dtype} b = {acc.id}[k] + a;
                            {tmp.id}[k] = ({tmp.dtype}) ((b - {acc.id}[k]) - a);
                            {acc.id}[k] = b;
                        }}
                """
            
    def DiffT(self, v, gradin, f0=None):
        return Sum_Reduction(Grad(self.formula,v,gradin),v.cat%2)
        

class Max_Reduction(Reduction):
    
    #// Implements the max reduction operation : for each i or each j, find the
    #// maximal value of Fij
    #// operation is vectorized: if Fij is vector-valued, max is computed for each dimension.
    
    string_id = "Max_Reduction"
    
    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = formula.dim                      # dimension of final output of reduction
        self.dimred = self.dim                      # dimension of inner reduction variables
        
    def InitializeReduction(self, acc):
        # Returns C++ code to be used at initialization phase of the reduction.
        # Here it consists in setting the output array to -infinity.
        return acc.assign(neg_infinity(acc.dtype))
        
    def ReducePairScalar(self, acc, xi):
        # Subroutine of ReducePairShort and ReducePair methods.
        if xi.dtype == "half2":
            raise ValueError("not implemented")
        return f"""
                    if({xi.id}>{acc.id}) 
                        {acc.id} = {xi.id};
                """

class Max_ArgMax_Reduction_Base(Reduction):
    
    #/////////////////////////////////////////////////////////////////////////
    #//          max+argmax reduction : base class                          //
    #/////////////////////////////////////////////////////////////////////////
    
    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        
        # We work with a (values,indices) vector
        self.dimred = 2*formula.dim                      # dimension of inner reduction variables
        
    def InitializeReduction(self, acc):
        # Returns C++ code to be used at initialization phase of the reduction.
        acc_max, acc_argmax = acc.split(self.dim, self.dim)
        return acc_max.assign(neg_infinity(acc.dtype)) + acc_argmax.assign(c_zero_float)
        
    def ReducePairScalar(self, acc_val, acc_ind, xi, ind):
        # Subroutine of ReducePairShort and ReducePair methods.
        if xi.dtype == "half2":
            raise ValueError("not implemented")
        return f"""
                    if({xi.id}>{acc_val.id}) 
                    {{
                        {acc_val.id} = {xi.id};
                        {acc_ind.id} = {ind.id};
                    }}
                """        

    def ReducePair(self, acc, xi):
        # Returns C++ code that implements the update phase of the reduction.
        acc_val, acc_ind = acc.split(self.dim, self.dim)     
        xi_val, xi_ind = xi.split(self.dim, self.dim)     
        return VectApply(self.ReducePairScalar, acc_val, xi_val, xi_ind)
        
    def ReducePairShort(self, acc, xi, ind):
        if xi.dtype == "half2":
            raise ValueError("not implemented")
            half2_val = c_variable("half2_ind")
            string = half2_val.declare_assign(f"__floats2half2_rn(2*{ind()},2*{ind()}+1)")
        acc_val, acc_ind = acc.split(self.dim, self.dim)    
        ind_arr = c_array(xi.dtype, 1, f"(&{ind.id})")
        return VectApply(self.ReducePairScalar, acc_val, acc_ind, xi, ind_arr)

class Max_ArgMax_Reduction(Max_ArgMax_Reduction_Base):
    
    #// Implements the max+argmax reduction operation : for each i or each j, find the maximal value of Fij and its index
    #// operation is vectorized: if Fij is vector-valued, max+argmax is computed for each dimension.
    
    string_id = "Max_ArgMax_Reduction"
    
    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = 2*formula.dim    
        
    def FinalizeOutput(self, acc, out, i):
        return VectCopy(out, acc) 
        
class ArgMax_Reduction(Max_ArgMax_Reduction_Base):
    
    #// Implements the argmax reduction operation : for each i or each j, find the index of the
    #// maximal value of Fij
    #// operation is vectorized: if Fij is vector-valued, argmax is computed for each dimension.
    
    string_id = "ArgMax_Reduction"
    
    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = formula.dim    
        
    def FinalizeOutput(self, acc, out, i):
        acc_val, acc_ind = acc.split(self.dim, self.dim)
        return VectCopy(out, acc_ind) 
        
    def DiffT(self, v, gradin):
        return Zero_Reduction(v.dim,v.cat%2)


class Min_Reduction(Reduction):
    
    #// Implements the min reduction operation : for each i or each j, find the
    #// minimal value of Fij
    #// operation is vectorized: if Fij is vector-valued, min is computed for each dimension.
    
    string_id = "Min_Reduction"
    
    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = formula.dim                      # dimension of final output of reduction
        self.dimred = self.dim                      # dimension of inner reduction variables
        
    def InitializeReduction(self, acc):
        # Returns C++ code to be used at initialization phase of the reduction.
        # Here it consists in setting the output array to -infinity.
        return acc.assign(infinity(acc.dtype))
        
    def ReducePairScalar(self, acc, xi):
        # Subroutine of ReducePairShort and ReducePair methods.
        if xi.dtype == "half2":
            raise ValueError("not implemented")
        return f"""
                    if({xi.id}<{acc.id}) 
                        {acc.id} = {xi.id};
                """

class Min_ArgMin_Reduction_Base(Reduction):
    
    #/////////////////////////////////////////////////////////////////////////
    #//          min+argmin reduction : base class                          //
    #/////////////////////////////////////////////////////////////////////////
    
    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        
        # We work with a (values,indices) vector
        self.dimred = 2*formula.dim                      # dimension of inner reduction variables
        
    def InitializeReduction(self, acc):
        # Returns C++ code to be used at initialization phase of the reduction.
        acc_min, acc_argmin = acc.split(self.dim, self.dim)
        return acc_min.assign(infinity(acc.dtype)) + acc_argmin.assign(c_zero_float)
        
    def ReducePairScalar(self, acc_val, acc_ind, xi, ind):
        # Subroutine of ReducePairShort and ReducePair methods.
        if xi.dtype == "half2":
            raise ValueError("not implemented")
        return f"""
                    if({xi.id}<{acc_val.id}) 
                    {{
                        {acc_val.id} = {xi.id};
                        {acc_ind.id} = {ind.id};
                    }}
                """        

    def ReducePair(self, acc, xi):
        # Returns C++ code that implements the update phase of the reduction.
        acc_val, acc_ind = acc.split(self.dim, self.dim)     
        xi_val, xi_ind = xi.split(self.dim, self.dim)     
        return VectApply(self.ReducePairScalar, acc_val, xi_val, xi_ind)
        
    def ReducePairShort(self, acc, xi, ind):
        if xi.dtype == "half2":
            raise ValueError("not implemented")
            half2_val = c_variable("half2_ind")
            string = half2_val.declare_assign(f"__floats2half2_rn(2*{ind()},2*{ind()}+1)")
        acc_val, acc_ind = acc.split(self.dim, self.dim)    
        ind_arr = c_array(xi.dtype, 1, f"(&{ind.id})")
        return VectApply(self.ReducePairScalar, acc_val, acc_ind, xi, ind_arr)

class Min_ArgMin_Reduction(Min_ArgMin_Reduction_Base):
    
    #// Implements the min+argmin reduction operation : for each i or each j, find the minimal value of Fij and its index
    #// operation is vectorized: if Fij is vector-valued, min+argmain is computed for each dimension.
    
    string_id = "Min_ArgMin_Reduction"
    
    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = 2*formula.dim    
        
    def FinalizeOutput(self, acc, out, i):
        return VectCopy(out, acc) 
        
class ArgMin_Reduction(Min_ArgMin_Reduction_Base):
    
    #// Implements the argmin reduction operation : for each i or each j, find the index of the
    #// minimal value of Fij
    #// operation is vectorized: if Fij is vector-valued, argmin is computed for each dimension.
    
    string_id = "ArgMin_Reduction"
    
    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = formula.dim    
        
    def FinalizeOutput(self, acc, out, i):
        acc_val, acc_ind = acc.split(self.dim, self.dim)
        return VectCopy(out, acc_ind) 
        
    def DiffT(self, v, gradin):
        return Zero_Reduction(v.dim,v.cat%2)


class KMin_ArgKMin_Reduction(Reduction):
    
    #// Implements the k-min-arg-k-min reduction operation : for each i or each j, find the values and indices of the
    #// k minimal values of Fij
    #// operation is vectorized: if Fij is vector-valued, arg-k-min is computed for each dimension.
    
    string_id = "KMin_ArgKMin_Reduction"
    
    def __init__(self, formula, K, tagIJ):
        super().__init__(formula, tagIJ)
        
        self.K = K
        
        # dim is dimension of output of reduction ; for a arg-k-min reduction it is equal to the dimension of output of formula
        self.dim = 2*K*formula.dim
        
        # We work with a (values,indices) vector
        self.dimred = self.dim                      # dimension of inner reduction variables
        
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

class ArgKMin_Reduction(KMin_ArgKMin_Reduction):
    
    #// Implements the arg-k-min reduction operation : for each i or each j, find the indices of the
    #// k minimal values of Fij
    #// operation is vectorized: if Fij is vector-valued, arg-k-min is computed for each dimension.
    
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
        return Zero_Reduction(v.dim,v.cat%2)

class KMin_Reduction(KMin_ArgKMin_Reduction):
    
    #// Implements the k-min reduction operation : for each i or each j, find the
    #// k minimal values of Fij
    #// operation is vectorized: if Fij is vector-valued, arg-k-min is computed for each dimension.
    
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


class Max_SumShiftExpWeight_Reduction(Reduction):
    
    #// Implements the coupled reduction operation m_i=max_j f_ij, s_i=sum_j exp(f_ij-m_i) g_ij
    #// where f and g are two formulas. f must be scalar-valued.
    #// This reduciton is the base for numerically stable computation of log-sum-exp and softmax type reductions.

    string_id = "Max_SumShiftExpWeight_Reduction"
    
    def __init__(self, formulaF, tagIJ, formulaG = IntCst(1)):
        if formulaF.dim != 1:
            raise ValueError("Max_SumShiftExpWeight_Reduction requires first formula of dimension 1.")
        super().__init__(Concat(formulaF, formulaG), tagIJ)
        self.formulaF = formulaF
        self.formulaG = formulaG
        self.dim = formulaF.dim + formulaG.dim       # dimension of final output of reduction
        self.dimred = self.dim                      # dimension of inner reduction variables
        
    def InitializeReduction(self, acc):
        # Returns C++ code to be used at initialization phase of the reduction.
        # We fill empty cells with the neutral element of the reduction operation,
        #                   (-inf,0) = e^{-inf} * 0 = 0
        m, s = acc.split(1, self.formulaG.dim)
        return m.assign(neg_infinity(acc.dtype)) + s.assign(c_zero_float)
        
    def ReducePair(self, acc, xi):
        # Returns C++ code that implements the update phase of the reduction.
        # (m,s) + (m',s'), i.e. exp(m)*s + exp(m')*s'
        
        if xi.dtype == "half2":
            raise ValueError("Not implemented.")
        
        return f"""       
                      {acc.dtype} tmpexp;
                      if ({acc.id}[0] > {xi.id}[0]) {{ // =  exp(m)  * (s + s'*exp(m'-m))   if m > m'
                        tmpexp = exp({xi.id}[0] - {acc.id}[0]);
                        #pragma unroll
                        for (int k = 1; k < {self.dimred}; k++)
                          {acc.id}[k] += {xi.id}[k] * tmpexp;
                      }} else {{             // =  exp(m') * (s' + exp(m-m')*s)   if m <= m'
                        tmpexp = exp({acc.id}[0] - {xi.id}[0]);
                        #pragma unroll
                        for (int k = 1; k < {self.dimred}; k++)
                          {acc.id}[k] = {xi.id}[k] + tmpexp * {acc.id}[k];
                        {acc.id}[0] = {xi.id}[0];
                      }}
              """
        
    def ReducePairShort(self, acc, xi, ind):
        return self.ReducePair(acc, xi)
        
    def KahanScheme(self, acc, xi, tmp):
        if xi.dtype == "half2":
            raise ValueError("Not implemented.")
        return f"""
                        {acc.id}.dtype tmpexp;
                        if ({acc.id}[0] > {xi.id}[0])    // =  exp(m)  * (s + s'*exp(m'-m))   if m > m'
                        {{      
                            tmpexp = exp({xi.id}[0] - {acc.id}[0]);
                            #pragma unroll
                        	for (int k=1; k<{self.dimred}; k++)
                            {{
                        		{acc.dtype} a = {xi.id}[k] * tmpexp - {tmp.id}[k-1];
                        		{acc.dtype} b = {acc.id}[k] + a;
                        		{tmp.id}[k-1] = (b - {acc.id}[k]) - a;
                        		{acc.id}[k] = b;
                        	}}
                        }} 
                        else      // =  exp(m') * (s' + exp(m-m')*s)   if m <= m'
                        {{             
                            tmpexp = exp({acc.id}[0] - {xi.id}[0]);
                            #pragma unroll
                            for (int k = 1; k < {self.dimred}; k++)
                            {{
                        		{acc.dtype} u = tmpexp * {acc.id}[k];
                        		{acc.dtype} a = {xi.id}[k] - tmpexp * {tmp.id}[k-1];
                        		{acc.dtype} b = u + a;
                        		{tmp.id}[k-1] = (b - u) - a;
                        		{acc.id}[k] = b;
                        	}}
                        	{acc.id}[0] = {xi.id}[0];
                        }}
                """
    
    def DiffT(self, v, gradin, MS):
        """
          // Beware: the formula that we use for the gradient is *only* valid
          // if the output [M,S] = Max_SumShiftExp(F,G) has been flattened through a
          // L = M + log(S) (Log-Sum-Exp) or a weighted Soft-Max
          // operation (as done by the Python bindings), and if 
          // GRADIN = [Grad(L), Grad(L)/S ]
          // has been backpropagated from L.
        """
        M = Extract(MS, 0, self.formulaF.dim)
        S = Extract(gradin, self.formulaF.dim, self.formulaG.dim)      
        return Grad( Sum_Reduction( Exp(self.formulaF-M)*self.formulaG, self.tagI ), v, S)
        
Max_SumShiftExp_Reduction = Max_SumShiftExpWeight_Reduction








#/////////////////////////////////////////////////////////////




#///      GRADIENT OPERATOR  : Grad< F, V, Gradin >       ////
#/////////////////////////////////////////////////////////////

# Defines [\partial_V F].gradin function
# Symbolic differentiation is a straightforward recursive operation,
# provided that the operators have implemented their DiffT "compiler methods":
def Grad(formula,v,gradin=None):
    if gradin==None:
        if v.cat==2:
            raise ValueError("not implemented")
        inds = GetInds(formula.Vars_)
        ind = 1 + max(inds) if len(inds)>0 else 0
        dim = formula.dim
        cat = 1-v.cat
        gradin = Var(ind,dim,cat) 
    return formula.DiffT(v,gradin)
        
# same with additional saved forward variable. This is only used for taking gradients of reductions operations.
def Grad_WithSavedForward(red_formula, v, gradin, f0):
    return red_formula.DiffT(v,gradin,f0)




class getReduction:
    library = {}
    def __new__(self, red_formula_string, aliases=[]):
        aliases_dict = {}
        for alias in aliases:
            if "=" in alias:
                varname, var = alias.split("=")
                aliases_dict[varname] = eval(var)
        string_id_hash = get_hash_name(red_formula_string, aliases)
        if string_id_hash in getReduction.library:
            return getReduction.library[string_id_hash]
        else:
            reduction = eval(red_formula_string, globals(), aliases_dict)
            getReduction.library[string_id_hash] = reduction
            return reduction
