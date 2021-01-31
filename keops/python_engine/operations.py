from tree import tree
from utils import *

class Operation(tree):
    
    # Base class for all keops building block operations in a formula
    
    def __init__(self, *args):
        # *args are other instances of Operation, they are the child operations of self
        self.children = args
        self.params = ()
        # The variables in the current formula is the union of the variables in the child operations.
        # Note that this requires implementing properly __eq__ and __hash__ methods in Var class
        self.Vars_ = set.union(*(arg.Vars_ for arg in args)) if len(args)>0 else set()
        
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
                arg = c_array(arg_name, out.dtype, child.dim)
                # Now we append into string the C++ code to declare the array
                string += f"{arg.declare()}\n"
                # Now we evaluate the child operation and append the result into string
                string += child(arg, table)    
            args.append(arg)
        # Finally, evaluation of the operation itself
        string += self.Op(out, table, *args)
        string += f"\n\n// Finished code block for {self.__repr__()}.\n}}\n\n"
        return string
            
    def __mul__(self, other):
        # f*g redirects to Mult(f,g)
        return Mult(self, other)
        
    def __rmul__(self, other):
        # g*f redirects to Mult(f,g)
        return Mult(self, other)
        
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
            print("not implemented")
            
            

        
        
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
        zero = c_variable("0.0f","float")
        return out.assign(zero)

    def DiffT(self,v,gradin):
        return Zero(v.dim)
    
class IntCst(Operation):
    # constant integer "operation"
    string_id = "IntCst"
    print_spec = "", "pre", 0
    
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.dim = 1
        self.params = (value,)
        
    # custom __eq__ method
    def __eq__(self, other):
        return type(self)==type(other) and self.value==other.value
        
    def Op(self, out, table):
        return f"*{out()} = {cast_to(value(out.dtype))}((float){self.value});\n"

    def DiffT(self,v,gradin):
        return Zero(v.dim)
        
        
        
    
class VectorizedScalarUnaryOp(Operation):
    # class for unary operations that are vectorized scalar operations,
    # such as Exp(f), Cos(f), etc.
    
    @property
    def dim(self):
        # dim gives the output dimension of the operation, 
        # here it is the same as the output dimension of the child operation
        return self.children[0].dim
        
    def Op(self, out, table, arg):
        # Atomic evaluation of the operation : it consists in a simple
        # for loop around the call to the correponding scalar operation
        return VectApply(self.ScalarOp, out, arg)
        

class VectorizedScalarBinaryOp(Operation):
    # class for binary operations that are vectorized or broadcasted scalar operations,
    # such as Mult(f), Subtract(f), etc.
    
    @property
    def dim(self):
        # dim gives the output dimension of the operation, 
        # here it is the max of the child operations (because we allow for broadcasting here)
        return max(self.children[0].dim, self.children[1].dim)
        
    def Op(self, out, table, arg0, arg1):
        # Atomic evaluation of the operation : it consists in a simple
        # for loop around the call to the correponding scalar operation
        return VectApply(self.ScalarOp, out, arg0, arg1)
        
class Exp(VectorizedScalarUnaryOp):
    # the exponential vectorized operation
    string_id = "Exp"
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = {keops_exp(arg)};\n"
    def DiffT(self,v,gradin):
        # [\partial_V exp(F)].gradin = exp(F) * [\partial_V F].gradin
        f = self.children[0]
        return f.DiffT(v,Exp(f)*gradin)
    
class Minus_(VectorizedScalarUnaryOp):
    # the "minus" vectorized operation
    string_id = "Minus"
    print_spec = "-", "pre", 2
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = -{arg()};\n"
    def DiffT(self,v,gradin):
        f = self.children[0]
        return -f.DiffT(v,gradin)

def Minus(arg):
    if isinstance(arg,Zero):
        return arg
    else:
        return Minus_(arg)
            
class Square_(VectorizedScalarUnaryOp):
    # the square vectorized operation
    string_id = "Square"
    print_spec = "**2", "post", 1
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = {arg()}*{arg()};\n"
    def DiffT(self,v,gradin):
        # [\partial_V (F)**2].gradin = F * [\partial_V F].gradin
        f = self.children[0]
        return IntCst(2) * f.DiffT(v, f*gradin)

def Square(arg):
    if isinstance(arg,Zero):
        return arg
    else:
        return Square_(arg)
                
class Sum_(Operation):
    # the summation operation
    string_id = "Sum"
    dim = 1
    def Op(self, out, table, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return out.assign(c_zero_float) + VectApply(self.ScalarOp, out, arg)
    def ScalarOp(self,out,arg):
        return f"{out()} += {arg()};\n"
    def DiffT(self,v,gradin):
        f = self.children[0]
        return f.DiffT(v,SumT(gradin,f.dim))

def Sum(arg):
    if isinstance(arg,Zero):
        return Zero(1)
    else:
        return Sum_(arg)

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
        value_arg = c_variable(f"*{arg()}",value(arg.dtype))
        return out.assign(value_arg)
    def DiffT(self,v,gradin):
        f = self.children[0]
        return f.DiffT(v,Sum(gradin))

def SumT(arg, dim):
    if isinstance(arg,Zero):
        return Zero(dim)
    else:
        return SumT_(arg,dim)    
            
class Mult_(VectorizedScalarBinaryOp):
    # the binary multiply operation
    string_id = "Mult" 
    print_spec = "*", "mid", 3       
    def ScalarOp(self,out,arg0,arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = {arg0()}*{arg1()};\n"
    #  \diff_V (A*B) = (\diff_V A) * B + A * (\diff_V B)    
    def DiffT(self,v,gradin):
        fa, fb = self.children
        return fa.DiffT(v,fb*gradin) + fb.DiffT(v,fa*gradin)

def Mult(arg0, arg1):
    if isinstance(arg0,Zero):
        return arg0
    elif isinstance(arg1,Zero):
        return arg1
    elif isinstance(arg1,int):
        return Mult_(IntCst(arg1),arg0)
    else:
        return Mult_(arg0, arg1)
        
class Add_(VectorizedScalarBinaryOp):
    # the binary addition operation
    string_id = "Add"
    print_spec = "+", "mid", 4
    def ScalarOp(self,out,arg0,arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = {arg0()}+{arg1()};\n"
    def DiffT(self,v,gradin):
        fa, fb = self.children
        return fa.DiffT(v,gradin) + fb.DiffT(v,gradin)

def Add(arg0, arg1):
    if isinstance(arg0,Zero):
        return arg1
    elif isinstance(arg1,Zero):
        return arg0
    elif arg0==arg1:
        return IntCst(2)*arg0
    else:
        return Add_(arg0, arg1)
        
class Subtract_(VectorizedScalarBinaryOp):
    # the binary subtract operation
    string_id = "Subtract"
    print_spec = "-", "mid", 4
    def ScalarOp(self,out,arg0,arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = {arg0()}-{arg1()};\n"
    def DiffT(self,v,gradin):
        fa, fb = self.children
        return fa.DiffT(v,gradin) - fb.DiffT(v,gradin)

def Subtract(arg0, arg1):
    if isinstance(arg0,Zero):
        return -arg1
    elif isinstance(arg1,Zero):
        return arg0
    else:
        return Subtract_(arg0, arg1)


