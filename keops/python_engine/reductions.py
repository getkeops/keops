from tree import tree
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

    def FinalizeOutput(self, acc, out, i):
        # Returns C++ code that implements the final output of the reduction.
        # For most reducitons it is a simple copy of the temporary variable
        # updated during the reduction, with possibly a cast if the accumulator was of
        # different data type.
        return VectCopy(out, acc)
    


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
        
    def ReducePairShort(self, tmp, xi, ind):
        # Returns C++ code that implements the update phase of the reduction.
        # Here for the sum reduction it consists in a vectorized version of the "+=" operation.
        
        # N.B next lines are useless for SumReduction, but to be used in other reductions :
        # if xi.dtype == "half2":
        #     half2_val = c_variable("half2_ind")
        #     string = half2_val.declare_assign(f"__floats2half2_rn(2*{ind()},2*{ind()}+1)")
        
        return VectApply(self.ReducePairScalar, tmp, xi)
        
    def ReducePair(self, acc, xi):
        # Returns C++ code that implements the update phase of the reduction.
        # Here for the sum reduction it consists in a vectorized version of the "+=" operation.
        return VectApply(self.ReducePairScalar, acc, xi)
        
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
        string = f"{acc.id}[0] = {neg_infinity(acc.dtype)};\n"
        v = c_array(f"({acc.id}+1)", acc.dtype, self.dimred-1)
        string += v.assign(c_zero_float)
        return string
        
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
