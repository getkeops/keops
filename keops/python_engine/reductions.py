from tree import tree
from operations import Var
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



class Sum_Reduction(Reduction):
    # Sum reduction class
    string_id = "Sum_Reduction"
    
    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = formula.dim                      # dimension of final output of reduction
        self.dimred = self.dim                      # dimension of inner reduction variables
        self.Vars_ = formula.Vars_
        self.dim_kahan = self.dim
        
    def InitializeReduction(self, tmp):
        # Returns C++ code to be used at initialization phase of the reduction.
        # Here it consists in setting the output array to zero.
        return tmp.assign(c_zero_float)
        
    def ReducePairScalar(self, tmp, xi):
        # Subroutine of ReducePairShort and ReducePair methods.
        # Returns C++ code that implements the "+=" accumulation operation of the sum reduction
        return f"{tmp()} += {cast_to(tmp.dtype)}({xi()});"
        
    def ReducePairShort(self, tmp, xi, val):
        # Returns C++ code that implements the update phase of the reduction.
        # Here for the sum reduction it consists in a vectorized version of the "+=" operation.
        return VectApply(self.ReducePairScalar, tmp, xi)
        
    def ReducePair(self, acc, xi):
        # Returns C++ code that implements the update phase of the reduction.
        # Here for the sum reduction it consists in a vectorized version of the "+=" operation.
        return VectApply(self.ReducePairScalar, acc, xi)
        
    def KahanScheme(self, acc, xi, tmp):
        string = ""
        string +=  "#pragma unroll\n"
        string += f"for (int k=0; k<{self.dim}; k++)\n"
        string +=  "{\n"
        string += f"  {acc.dtype} a = {cast_to(acc.dtype)}({xi()}[k] - {tmp()}[k]);\n"
        string += f"  {acc.dtype} b = {acc()}[k] + a;\n"
        string += f"  {tmp()}[k] = {cast_to(tmp.dtype)}((b - {acc()}[k]) - a);\n"
        string += f"  {acc()}[k] = b;\n"
        string +=  "}\n"
        return string
        
    def FinalizeOutput(self, acc, out, i):
        # Returns C++ code that implements the final output of the reduction.
        # Here for the sum reduction it is a simple copy of the temporary variable
        # updated during the reduction, with possibly a cast if the accumulator was of
        # different data type.
        return VectCopy(out, acc)
    
    def DiffT(self, v, gradin, f0=None):
        return Sum_Reduction(Grad(self.formula,v,gradin),v.cat%2)
        


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
