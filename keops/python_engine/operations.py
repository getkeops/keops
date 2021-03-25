from tree_class import tree
from utils import *

###################
## Base class
###################

class Operation(tree):
    
    # Base class for all keops building block operations in a formula
    
    def __init__(self, *args):
        # *args are other instances of Operation, they are the child operations of self
        self.children = args
        self.params = ()
        # The variables in the current formula is the union of the variables in the child operations.
        # Note that this requires implementing properly __eq__ and __hash__ methods in Var class
        self.Vars_ = set.union(*(arg.Vars_ for arg in args)) if len(args)>0 else set()
        self.size_i = 
        
    def Vars(self, cat="all"):
        # if cat=="all", returns the list of all variables in a formula, stored in self.Vars_
        # if cat is an integer between 0 and 2, returns the list of variables v such that v.cat=cat
        if cat=="all":
            return list(self.Vars_)
        else:
            res = []
            for v in self.Vars_:
                if v.cat == cat:
                    res.append(v)
            return res
            
    def __call__(self, out, table):
        # returns the C++ code string corresponding to the evaluation of the formula
        # - out is a c_variable in which the result of the evaluation is stored
        # - table is the list of c_variables corresponding to actual local variables
        # required for evaluation : each Var(ind,*,*) corresponds to table[ind]
        string = f"\n{{\n// Starting code block for {self.__repr__()}.\n\n"
        args = []
        # Evaluation of the child operations
        for child in self.children:
            if isinstance(child,Var):
                # if the child of the operation is a Var, we do not need to evaluate it,
                # we simply record the corresponding c_variable
                arg = table[child.ind]
            else:
                # otherwise, we need to evaluate the child operation.
                # We first create a new c_array to store the result of the child operation.
                # This c_array must have a unique name in the code, to avoid conflicts
                # when we will recursively evaluate nested operations.
                template_string_id = "out_" + child.string_id.lower()
                arg_name = new_c_varname(template_string_id)
                arg = c_array(out.dtype, child.dim, arg_name)
                # Now we append into string the C++ code to declare the array
                string += f"{arg.declare()}\n"
                # Now we evaluate the child operation and append the result into string
                string += child(arg, table)    
            args.append(arg)
        # Finally, evaluation of the operation itself
        string += self.Op(out, table, *args)
        string += f"\n\n// Finished code block for {self.__repr__()}.\n}}\n\n"
        return string
        
    def Grad(self, v, gradin):
        if gradin.dim != self.dim:
            raise ValueError("incompatible dimensions")
        return self.DiffT(v, gradin)
            
    def __mul__(self, other):
        # f*g redirects to Mult(f,g)
        return Mult(self, other)
        
    def __rmul__(self, other):
        # g*f redirects to Mult(f,g)
        return Mult(self, other)
        
    def __truediv__(self, other):
        # f/g redirects to Divide(f,g)
        return Divide(self, other)
        
    def __add__(self, other):
        # f+g redirects to Add(f,g)
        return Add(self, other)
            
    def __sub__(self, other):
        # f-g redirects to Subtract(f,g)
        return Subtract(self, other)
            
    def __neg__(self):
        # -f redirects to Minus(f)
        return Minus(self)
        
    def __pow__(self, other):
        # f**2 redirects to Square(f)
        if other==2:
            return Square(self)
        else:
            raise ValueError("not implemented")
            
    def __or__(self, other):
        # f|g redirects to Scalprod(f,g)
        return Scalprod(self, other)
            
            
class VectorizedScalarOp(Operation):
    # class for operations that are vectorized or broadcasted
    # scalar operations,
    # such as Exp(f), Cos(f), Mult(f,g), Subtract(f,g), etc.
    
    def __init__(self, *args):
        dims = set(arg.dim for arg in args)
        if len(dims)>2 or (len(dims)==2 and min(dims)!=1):
            raise ValueError("dimensions are not compatible for VectorizedScalarOp")
        super().__init__(*args)
    
    @property
    def dim(self):
        # dim gives the output dimension of the operation, 
        # here it is the same as the output dimension of the child operation
        return max(child.dim for child in self.children)
        
    def Op(self, out, table, *arg):
        # Atomic evaluation of the operation : it consists in a simple
        # for loop around the call to the correponding scalar operation
        return VectApply(self.ScalarOp, out, *arg)



#######################
## Var operation
#######################    
        
class Var(Operation):
    # Var operation class. Var(ind,dim,cat) is a symbolic
    # object that encodes an input tensor in the call to the
    # KeOps routine, where
    # - ind gives the position of the input tensor in the list of tensors sent to the routine
    # - dim gives the "dimension" of the data : each input tensor is interpreted as a matrix
    # of size (n,dim), where n is dynamically handled and dim is known at compile time.
    # - cat is the "category" of the variable : either a "i"-indexed variable (cat=0), 
    # a "j"-indexed variable (cat=1), or a parameter variable (cat=2)
    
    string_id = "Var"
    
    def __init__(self, ind, dim, cat):
        super().__init__()
        self.ind = ind
        self.dim = dim
        self.cat = cat
        self.Vars_ = {self}
        self.params = (ind, dim, cat)
        
    # custom __eq__ and __hash__ methods, required to handle properly the union of two sets of Var objects
    def __eq__(self, other):
        return type(self)==type(other) and self.ind == other.ind and self.dim == other.dim and self.cat == other.cat 
    def __hash__(self):
        return hash((self.ind,self.dim,self.cat))
        
    def Op(self, out, table):
        return VectCopy(out, table[self.ind], cast=False)    
    
    # Assuming that the gradient wrt. Var is GRADIN, how does it affect V ?
    # Var::DiffT<V, grad_input> = grad_input   if V == Var (in the sense that it represents the same symb. var.)
    #                             Zero(V::DIM) otherwise
    def DiffT(self,v,gradin):
        return gradin if v==self else Zero(v.dim)
    




###################
## Constants
###################


class Zero(Operation):
    # zero operation : encodes a vector of zeros
    string_id = "Zero"
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.params = (dim,)
    
    # custom __eq__ method
    def __eq__(self, other):
        return type(self)==type(other) and self.dim==other.dim
        
    def Op(self, out, table):
        zero = c_variable("float", "0.0f")
        return out.assign(zero)

    def DiffT(self,v,gradin):
        return Zero(v.dim)
    
class IntCst(Operation):
    # constant integer "operation"
    string_id = "IntCst"
    print_spec = "", "pre", 0
    
    def __init__(self, val):
        super().__init__()
        self.val = val
        self.dim = 1
        self.params = (val,)
        
    # custom __eq__ method
    def __eq__(self, other):
        return type(self)==type(other) and self.val==other.val
        
    def Op(self, out, table):
        return f"*{out.id} = {cast_to(out.dtype)}((float){self.val});\n"

    def DiffT(self,v,gradin):
        return Zero(v.dim)
        
        
        
    

##################################################
##########                          ##############
##########   Basic math operators   ##############
##########                          ##############
##################################################



##########################
######    Add        #####
##########################

class Add_(VectorizedScalarOp):
    # the binary addition operation
    string_id = "Add"
    print_spec = "+", "mid", 4
    def ScalarOp(self,out,arg0,arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {arg0.id}+{arg1.id};\n"
    def DiffT(self,v,gradin):
        fa, fb = self.children
        return fa.Grad(v,gradin) + fb.Grad(v,gradin)

def Add(arg0, arg1):
    if isinstance(arg0,Zero):
        return Broadcast(arg1, arg0.dim)
    elif isinstance(arg1,Zero):
        return Broadcast(arg0, arg1.dim)
    elif arg0==arg1:
        return IntCst(2)*arg0
    else:
        return Add_(arg0, arg1)



##########################
######    Subtract   #####
##########################

class Subtract_(VectorizedScalarOp):
    # the binary subtract operation
    string_id = "Subtract"
    print_spec = "-", "mid", 4
    def ScalarOp(self,out,arg0,arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {arg0.id}-{arg1.id};\n"
    def DiffT(self,v,gradin):
        fa, fb = self.children
        return fa.Grad(v,gradin) - fb.Grad(v,gradin)        
        
def Subtract(arg0, arg1):
    if isinstance(arg0,Zero):
        return -Broadcast(arg1, arg0.dim)
    elif isinstance(arg1,Zero):
        return Broadcast(arg0, arg1.dim)
    else:
        return Subtract_(arg0, arg1)
        
        

##########################
######    Minus      #####
##########################

class Minus_(VectorizedScalarOp):
    # the "minus" vectorized operation
    string_id = "Minus"
    print_spec = "-", "pre", 2
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = -{arg.id};\n"
    def DiffT(self,v,gradin):
        f = self.children[0]
        return -f.Grad(v,gradin)
        
def Minus(arg):
    if isinstance(arg,Zero):
        return arg
    else:
        return Minus_(arg)



##########################
######    Mult       #####
##########################

class Mult_(VectorizedScalarOp):
    # the binary multiply operation
    string_id = "Mult" 
    print_spec = "*", "mid", 3       
    def ScalarOp(self,out,arg0,arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {arg0.id}*{arg1.id};\n"
    #  \diff_V (A*B) = (\diff_V A) * B + A * (\diff_V B)    
    def DiffT(self,v,gradin):
        fa, fb = self.children
        if fa.dim == 1 and fb.dim > 1:
            return fa.Grad(v, Scalprod(gradin,fb)) + fb.Grad(v, fa*gradin)
        elif fb.dim == 1 and fa.dim > 1:
            return fa.Grad(v, fb*gradin) + fb.Grad(v, Scalprod(gradin,fa))
        else:
            return fa.Grad(v, fb*gradin) + fb.Grad(v, fa*gradin)
    
def Mult(arg0, arg1):
    if isinstance(arg0,Zero):
        return Broadcast(arg0, arg1.dim)
    elif isinstance(arg1,Zero):
        return Broadcast(arg1, arg0.dim)
    elif isinstance(arg1,int):
        return Mult(IntCst(arg1),arg0)
    else:
        return Mult_(arg0, arg1)
        


##########################
######    Divide     #####
##########################
        
class Divide_(VectorizedScalarOp):
    # the binary divide operation
    string_id = "Divide" 
    print_spec = "/", "mid", 3       
    def ScalarOp(self,out,arg0,arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {arg0.id} / {arg1.id};\n"
    #  \diff_V (A/B) = ((\diff_V A) * B - A * (\diff_V B)) / B^2
    def DiffT(self,v,gradin):
        fa, fb = self.children
        if fa.dim == 1 and fb.dim > 1:
            return (fa.Grad(v, Scalprod(gradin,fb)) - fb.Grad(v, fa*gradin)) / Square(fb)
        elif fb.dim == 1 and fa.dim > 1:
            return (fa.Grad(v, fb*gradin) - fb.Grad(v, Scalprod(gradin,fa))) / Square(fb)
        else:
            return (fa.Grad(v, fb*gradin) - fb.Grad(v, fa*gradin)) / Square(fb)
    
def Divide(arg0, arg1):
    if isinstance(arg0,Zero):
        return Broadcast(arg0, arg1.dim)
    elif isinstance(arg1,Zero):
        raise ValueError("division by zero")
    elif isinstance(arg1,int):
        return Divide(arg0, IntCst(arg1))
    else:
        return Divide_(arg0, arg1)
        
        
        
        
        
        
        
##################################################
##########                          ##############
##########      Math operations     ##############
##########                          ##############
##################################################



##########################
######    Square     #####
##########################

class Square_(VectorizedScalarOp):
    # the square vectorized operation
    string_id = "Square"
    print_spec = "**2", "post", 1
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {arg.id}*{arg.id};\n"
    def DiffT(self,v,gradin):
        # [\partial_V (F)**2].gradin = F * [\partial_V F].gradin
        f = self.children[0]
        return IntCst(2) * f.Grad(v, f*gradin)

def Square(arg):
    if isinstance(arg,Zero):
        return arg
    else:
        return Square_(arg)



##########################
######    Abs        #####
##########################
        
class Abs(VectorizedScalarOp):
    # the absolute value vectorized operation
    string_id = "Abs"
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {keops_abs(arg)};\n"
    def DiffT(self,v,gradin):
        f = self.children[0]
        return f.Grad(v,Sign(f)*gradin)
    
    

##########################
######    Exp        #####
##########################
        
class Exp(VectorizedScalarOp):
    # the exponential vectorized operation
    string_id = "Exp"
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {keops_exp(arg)};\n"
    def DiffT(self,v,gradin):
        # [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
        f = self.children[0]
        return f.Grad(v,Exp(f)*gradin)
    
    

##########################
######    Sqrt       #####
##########################
        
class Sqrt(VectorizedScalarOp):
    # the square root vectorized operation
    string_id = "Sqrt"
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {keops_sqrt(arg)};\n"
    def DiffT(self,v,gradin):
        # [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
        f = self.children[0]
        return f.Grad(v, IntInv(2) * Rsqrt(f) * gradin)
    
    

##########################
######    Acos       #####
##########################
        
class Acos(VectorizedScalarOp):
    # the arc-cosine vectorized operation
    string_id = "Acos"
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out.id} = {keops_acos(arg)};\n"
    def DiffT(self,v,gradin):
        f = self.children[0]
        return f.Grad(v, - Rsqrt( (IntCst(1)-Square(f)) ) * gradin)

    

##########################
######    Sum        #####
##########################            
                
class Sum_(Operation):
    # the summation operation
    string_id = "Sum"
    dim = 1
    def Op(self, out, table, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(c_zero_float) + VectApply(self.ScalarOp, out, arg)
    def ScalarOp(self,out,arg):
        return f"{out.id} += {arg.id};\n"
    def DiffT(self,v,gradin):
        f = self.children[0]
        return f.Grad(v,SumT(gradin,f.dim))

def Sum(arg):
    if isinstance(arg,Zero):
        return Zero(1)
    else:
        return Sum_(arg)



##########################
######    SumT       #####
##########################

class SumT_(Operation):
    # the adjoint of the summation operation
    string_id = "SumT"
    def __init__(self, arg, dim):
        super().__init__(arg)
        self.dim = dim
        self.params = (dim,)
    def __eq__(self, other):
        return type(self)==type(other) and self.dim==other.dim
    def Op(self, out, table, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        value_arg = c_variable(arg.dtype, f"*{arg.id}")
        return out.assign(value_arg)
    def DiffT(self,v,gradin):
        f = self.children[0]
        return f.Grad(v,Sum(gradin))

def SumT(arg, dim):
    if arg.dim != 1:
        raise ValueError("dimension of argument must be 1 for SumT operation")
    elif isinstance(arg,Zero):
        return Zero(dim)
    else:
        return SumT_(arg, dim)
        
                  





##########################
#####    Broadcast    ####
##########################        
         
# N.B. this is used internally 
def Broadcast(arg, dim):
    if arg.dim == dim or dim == 1:
        return arg
    elif arg.dim == 1:
        return SumT(arg, dim)
    else:
        raise ValueError("dimensions are not compatible for Broadcast operation")
    
    
    
############################
######    Concat       #####
############################

class Concat(Operation):
    string_id = "Concat"
    def __init__(self, arg0, arg1):
        super().__init__(arg0, arg1)
        self.dim = arg0.dim + arg1.dim
    def Op(self, out, table, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        out0, out1 = out.split(arg0.dim, arg1.dim)
        return VectCopy(out0, arg0) + VectCopy(out1, arg1)
    def DiffT(self,v,gradin):
        f = self.children[0]
        g = self.children[1]
        return f.Grad(v,Extract(gradin, 0, f.dim)) + g.Grad(v,Extract(gradin, f.dim, g.dim))
        
        
        
#//////////////////////////////////////////////////////////////
#////     VECTOR EXTRACTION : Extract<F,START,DIM>         ////
#//////////////////////////////////////////////////////////////

class Extract(Operation):
    string_id = "Extract"
    def __init__(self, arg0, start, dim):
        if arg0.dim < start+dim or start<0:
            raise ValueError("Index out of bound in Extract")
        super().__init__(arg0)
        self.start = start
        self.dim = dim
        self.params = (start, dim)
    def Op(self, out, table, arg0):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        v = c_array(arg0.dtype, out.dim, f"({arg0.id}+{self.start})")
        return VectCopy(out, v)
    def DiffT(self,v,gradin):
        f = self.children[0]
        return f.Grad(v,ExtractT(gradin, self.start, f.dim))     



#//////////////////////////////////////////////////////////////
#////     VECTOR "INJECTION" : ExtractT<F,START,DIM>       ////
#//////////////////////////////////////////////////////////////

class ExtractT(Operation):
    string_id = "ExtractT"
    def __init__(self, F, start, dim):
        if start+F.dim > dim or start<0:
            raise ValueError("Index out of bound in ExtractT")
        super().__init__(F)
        self.start = start
        self.dim = dim
        self.params = (start, dim)
        self.dimarg = F.dim
    def Op(self, out, table, arg0):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        out_prev, out_mid, out_end = out.split(self.start, self.dim, self.dimarg-self.start-self.dim)
        return "\n".join( 
                            out_prev.assign(c_zero_float),
                            VectCopy(out_mid, arg0),
                            out_end.assign(c_zero_float)
                        )
    def DiffT(self,v,gradin):
        f = self.children[0]
        return f.Grad(v,Extract(gradin, self.start, f.dim))     




##################################################
#######                                ###########
#######     Norm related operations    ###########
#######                                ###########
##################################################



##########################
######    Norm2      #####
##########################        
          
def Norm2(arg):
    return Sqrt(Scalprod(arg,arg))



##########################
####    Normalize    #####
##########################        
          
def Normalize(arg):
    return Rsqrt(SqNorm2(arg)) * arg



##########################
#####    Scalprod     ####
##########################        
          
def Scalprod(arg0, arg1):
    if arg0.dim == 1:
        return arg0 * Sum(arg1)
    elif arg1.dim == 1:
        return Sum(arg0) * arg1
    else:
        return Sum(arg0*arg1)



##########################
######    SqDist     #####
##########################        
          
def SqDist(arg0, arg1):
    return SqNorm2(arg0-arg1)



##########################
######    SqNorm2    #####
##########################        
              
def SqNorm2(arg0):
    return Scalprod(arg0,arg0)



#############################
######    SqNormDiag    #####
#############################

# Anisotropic (but diagonal) norm, if S.dim == A.dim:
# SqNormDiag(S,A) = sum_i s_i*a_i*a_i        
              
def SqNormDiag(S, A):
    return Sum( S * Square(A))



###############################################################
######     ISOTROPIC NORM : SqNormIso(S,A)    #################
###############################################################      
              
def SqNormIso(S, A):
    return S * SqNorm2(A)



###########################################################################
######   WEIGHTED SQUARED DISTANCE : WeightedSqDist(S,A)    ###############
###########################################################################      
              
def WeightedSqDist(S, A):
    return S * SqNorm2(A)



###########################################################################
####       Fully anisotropic norm, if S.dim == A.dim * A.dim          #####
###########################################################################

# SymSqNorm(A,X) = sum_{ij} a_ij * x_i*x_j
def SymSqNorm(A,X):
    return Sum( A * TensorProd(X,X) )

# WeightedSqNorm(A,X) : redirects to SqNormIso, SqNormDiag or SymSqNorm
# depending on dimension of A.

def WeightedSqNorm(A, X):
    if A.dim == 1:
        return SqNormIso(A,X)
    elif A.dim == X.dim:
        return SqNormDiag(A,X)
    else:
        return SymSqNorm(A,X)





##################################################
#######                                ###########
#######     Tensor operations          ###########
#######                                ###########
##################################################



#/////////////////////////////////////////////////////////////////////////
#////     Vector-matrix product           b x A                       ////
#/////////////////////////////////////////////////////////////////////////

class VecMatMult(Operation):
    string_id = "VecMatMult"
    def __init__(self, B, A):
        # A is vector of size n*p, interpreted as matrix, B is vector of size n, interpreted as row vector
        # output is vector of size p
        if A.dim % B.dim != 0:
            raise ValueError("Dimensions of A and B are not compatible for vector-matrix product")
        super().__init__(B, A)
        self.dim = A.dim // B.dim
    def Op(self, out, table, inB, inA):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"""
                    #if C_CONTIGUOUS     // row major
                        #pragma unroll
                        for (int i = 0; i < {self.dim}; i++) 
                        {{
                            {out.id}[i] = ({out.dtype})(0.0f);
                        	#pragma unroll
                            for (int k = 0; k < {inB.dim}; k++)
                                {out.id}[i] += {inA.id}[{self.dim} * k + i] * {inB.id}[k];
                        }}
                    #else               // column major
                        int q = 0;
                        #pragma unroll
                        for (int i = 0; i < {self.dim}; i++) 
                        {{
                            {out.id}[i] = ({out.dtype})(0.0f);
                            #pragma unroll
                            for (int k = 0; k < {inB.dim}; k++, q++)
                                {out.id}[i] += {inA.id}[q] * {inB.id}[k];
                        }}
                    #endif
                """
    def DiffT(self,v,gradin):
        B = self.children[0]
        A = self.children[1]
        return A.Grad(v,TensorProd(B, gradin)) + B.Grad(v,MatVecMult(A, gradin))



#/////////////////////////////////////////////////////////////////////////
#////     Matrix-vector product      A x b                           ////
#/////////////////////////////////////////////////////////////////////////

class MatVecMult(Operation):
    string_id = "MatVecMult"
    def __init__(self, A, B):
        # A is vector of size n*p, interpreted as matrix, B is vector of size p, interpreted as column vector
        # output is vector of size n
        if A.dim % B.dim != 0:
            raise ValueError("Dimensions of A and B are not compatible for matrix-vector product")
        super().__init__(A, B)
        self.dim = A.dim // B.dim
    def Op(self, out, table, inB, inA):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"""
                    #if C_CONTIGUOUS     // row major
                        int q = 0;
                        #pragma unroll
                        for (int i = 0; i < {self.dim}; i++) 
                        {{
                            {out.id}[i] = ({out.dtype})(0.0f);
                            #pragma unroll
                            for (int k = 0; k < {inB.dim}; k++, q++)
                                {out.id}[i] += {inA.id}[q] * {inB.id}[k];
                        }}
                    #else               // column major
                        #pragma unroll
                        for (int i = 0; i < {self.dim}; i++) 
                        {{
                            {out.id}[i] = ({out.dtype})(0.0f);
                        	#pragma unroll
                            for (int k = 0; k < {inB.dim}; k++)
                                {out.id}[i] += {inA.id}[{self.dim} * k + i] * {inB.id}[k];
                        }}
                    #endif
                """
    def DiffT(self,v,gradin):
        A = self.children[0]
        B = self.children[1]
        return A.Grad(v,TensorProd(gradin, B)) + B.Grad(v,VecMatMult(gradin, A))
        
        
        
####################################
######    Tensor product       #####
####################################

class TensorProd(Operation):
    string_id = "TensorProd"
    def __init__(self, arg0, arg1):
        super().__init__(arg0, arg1)
        self.dim = arg0.dim * arg1.dim
    def Op(self, out, table, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"""
                    #if C_CONTIGUOUS     // row major
                        int q = 0;
        	            #pragma unroll
                        for (int k = 0; k < {arg0.dim}; k++) 
                        {{
                            #pragma unroll
                            for (int l = 0; l < {arg1.dim}; l++, q++)
                                {out.id}[q] = {arg0.id}[k] * {arg1.id}[l];
                        }}
                    #else               // column major
                        int q = 0;
                        #pragma unroll
                        for (int i = 0; i < {arg0.dim}; i++) 
                        {{
                            #pragma unroll
                            for (int j = 0; j < {arg1.dim}; j++, q++)
                                {out.id}[{arg0.dim} * j + i] = {arg0.id}[i] * {arg1.id}[j];
                        }}
                    #endif
                """
    def DiffT(self,v,gradin):
        f = self.children[0]
        g = self.children[1]
        return f.Grad(v,MatVecMult(gradin, g)) + g.Grad(v,VecMatMult(f, gradin))


