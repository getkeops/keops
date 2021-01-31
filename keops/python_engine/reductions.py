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

          
class Var_loader:
    
    def __init__(self, red_formula):
        
        formula = red_formula.formula
        tagI, tagJ = red_formula.tagI, red_formula.tagJ
        
        self.Varsi = formula.Vars(cat=tagI)         # list all "i"-indexed variables in the formula
        self.indsi = GetInds(self.Varsi)            # list indices of "i"-indexed variables
        self.dimsx = GetDims(self.Varsi)            # list dimensions of "i"-indexed variables
        self.dimx = sum(self.dimsx)                 # total dimension of "i"-indexed variables
        
        self.Varsj = formula.Vars(cat=tagJ)         # list all "j"-indexed variables in the formula
        self.indsj = GetInds(self.Varsj)            # list indices of "j"-indexed variables
        self.dimsy = GetDims(self.Varsj)            # list dimensions of "j"-indexed variables
        self.dimy = sum(self.dimsy)                 # total dimension of "j"-indexed variables
        
        self.Varsp = formula.Vars(cat=2)            # list all parameter variables in the formula
        self.indsp = GetInds(self.Varsp)            # list indices of parameter variables
        self.dimsp = GetDims(self.Varsp)            # list indices of parameter variables
        self.dimp = sum(self.dimsp)                 # total dimension of parameter variables
        
        self.inds = GetInds(formula.Vars_)
        self.nminargs = max(self.inds)+1

    def table(self, xi, yj, pp):    
        table = [None] * self.nminargs
        for (dims, inds, xloc) in ((self.dimsx, self.indsi, xi), (self.dimsy, self.indsj, yj), (self.dimsp, self.indsp, pp)):
            k = 0
            for u in range(len(dims)):
                table[inds[u]] = c_array(f"({xloc()}+{k})", xloc.dtype, dims[u])
                k += dims[u]
        return table
    
    def direct_table(self, args, i, j):    
        table = [None] * self.nminargs
        for (dims, inds, row_index) in ((self.dimsx, self.indsi, i), (self.dimsy, self.indsj, j), (self.dimsp, self.indsp, c_zero_int)):
            for u in range(len(dims)):
                arg = args[inds[u]]
                table[inds[u]] = c_array(f"({arg()}+{row_index()}*{dims[u]})", value(arg.dtype), dims[u])
        return table
    
    def load_vars(self, cat, xloc, args, row_index=c_zero_int):
        # returns a c++ code used to create a local copy of slices of the input tensors, for evaluating a formula
        # cat is either "i", "j" or "p", specifying the category of variables to be loaded
        # - xloc is a c_array, the local array which will receive the copy
        # - args is a list of c_variable, representing pointers to input tensors 
        # - row_index is a c_variable (of dtype="int"), specifying which row of the matrix should be loaded
        #
        # Example: assuming i=c_variable("5","int"), xloc=c_variable("xi","float") and px=c_variable("px","float**"), then 
        # if self.dimsx = [2,2,3] and self.indsi = [7,9,8], the call to
        #   load_vars ( "i", xi, [arg0, arg1,..., arg9], row_index=i )
        # will output the following code:
        #   xi[0] = arg7[5*2+0];
        #   xi[1] = arg7[5*2+1];
        #   xi[3] = arg9[5*2+0];
        #   xi[4] = arg9[5*2+1];
        #   xi[5] = arg8[5*3+0];
        #   xi[6] = arg8[5*3+1];
        #   xi[7] = arg8[5*3+2];
        
        if cat=="i":
            dims, inds  = self.dimsx, self.indsi
        elif cat=="j":
            dims, inds = self.dimsy, self.indsj
        elif cat=="p":
            dims, inds = self.dimsp, self.indsp
        string = ""
        k = 0
        for u in range(len(dims)):
            for v in range(dims[u]):
                string += f"{xloc()}[{k}] = {args[inds[u]]()}[{row_index()}*{dims[u]}+{v}];\n"
                k+=1
        return string    
    


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
