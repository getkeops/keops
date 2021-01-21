from tree import tree
from utils import GetDims, GetInds, c_variable, VectAssign, VectApply, VectCopy, cast_to

class Reduction(tree):
    
    # Base class for all KeOps final reductions over a formula
    
    def __init__(self, formula, tagI):
        # - formula is an object of type Operation, it is the formula on which we apply a reduction
        # - tagI : 0 or 1, specifies wether we do the reduction over "i"-indexed or "j"-indexed variables.
        
        # We initialize several constants, most of them infered from the formula
        self.formula = formula
        self.children = [formula]
        self.tagI = tagI
        self.tagJ = 1-tagI
        self.cat = tagI
        
        self.Varsi = formula.Vars(cat=tagI)         # list all "i"-indexed variables in the formula
        self.indsi = GetInds(self.Varsi)            # list indices of "i"-indexed variables
        self.dimsx = GetDims(self.Varsi)            # list dimensions of "i"-indexed variables
        self.dimx = sum(self.dimsx)                 # total dimension of "i"-indexed variables
        
        self.Varsj = formula.Vars(cat=self.tagJ)    # list all "j"-indexed variables in the formula
        self.indsj = GetInds(self.Varsj)            # list indices of "j"-indexed variables
        self.dimsy = GetDims(self.Varsj)            # list dimensions of "j"-indexed variables
        self.dimy = sum(self.dimsy)                 # total dimension of "j"-indexed variables
        
        self.Varsp = formula.Vars(cat=2)            # list all parameter variables in the formula
        self.indsp = GetInds(self.Varsp)            # list indices of parameter variables
        self.dimsp = GetDims(self.Varsp)            # list indices of parameter variables
        self.dimp = sum(self.dimsp)                 # total dimension of parameter variables



class Sum_Reduction(Reduction):
    # Sum reduction class
    string_id = "Sum_Reduction"
    
    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = formula.dim                      # dimension of final output of reduction
        self.dimred = self.dim                      # dimension of inner reduction variables
        
    def InitializeReduction(self, tmp):
        # Returns C++ code to be used at initialization phase of the reduction.
        # Here it consists in setting the output array to zero.
        zero = c_variable("0.0f","float")
        return VectAssign(tmp, self.dim, zero)
        
    def ReducePairScalar(self, tmp, xi):
        # Subroutine of ReducePairShort and ReducePair methods.
        # Returns C++ code that implements the "+=" accumulation operation of the sum reduction
        return f"{tmp()} += {cast_to(tmp.dtype)}({xi()});"
        
    def ReducePairShort(self, tmp, xi, val):
        # Returns C++ code that implements the update phase of the reduction.
        # Here for the sum reduction it consists in a vectorized version of the "+=" operation.
        return VectApply(self.ReducePairScalar,self.dim, self.dim, tmp, xi)
        
    def FinalizeOutput(self, acc, out, i):
        # Returns C++ code that implements the final output of the reduction.
        # Here for the sum reduction it is a simple copy of the temporary variable
        # updated during the reduction, with possibly a cast if the accumulator was of
        # different data type.
        return VectCopy(self.dim, out, acc)
    
    def DiffT(self, v, gradin, f0=None):
        return Sum_Reduction(Grad(self.formula,v,gradin),v.cat%2)
        


#/////////////////////////////////////////////////////////////
#///      GRADIENT OPERATOR  : Grad< F, V, Gradin >       ////
#/////////////////////////////////////////////////////////////

# Defines [\partial_V F].gradin function
# Symbolic differentiation is a straightforward recursive operation,
# provided that the operators have implemented their DiffT "compiler methods":
def Grad(formula,v,gradin):
    return formula.DiffT(v,gradin)
        
# same with additional saved forward variable. This is only used for taking gradients of reductions operations.
def Grad_WithSavedForward(red_formula, v, gradin, f0):
    return red_formula.DiffT(v,gradin,f0)
