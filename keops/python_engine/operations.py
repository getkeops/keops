from tree import tree
from utils import new_c_variable, VectApply, VectCopy, VectApply2, value, keops_exp, cast_to

class Operation(tree):
    
    # Base class for all keops building block operations in a formula
    
    def Vars(self, cat="all"):
        # if cat=="all", returns the list of all variables in a formula, stored in self._Vars
        # if cat is an integer between 0 and 2, returns the list of variables v such that v.cat=cat
        if cat=="all":
            return list(self.Vars_)
        else:
            res = []
            for v in self._Vars:
                if v.cat == cat:
                    res.append(v)
            return res
            
    def __call__(self, out, table):
        # returns the C++ code string for the evaluation of the formula.
        # see Eval method implementations in UnaryOp, BinaryOp, etc.
        # for explanations.
        # Here we simply surround the piece of code with some C++ comments
        # for clarity of the output.
        string  = f"\n{{\n// Starting code block for {self.__repr__()}.\n\n"
        string += self.Eval(out, table)
        string += f"\n\n// Finished code block for {self.__repr__()}.\n}}\n\n"
        return string
            
    def __mul__(self, other):
        # f*g redirects to Mult(f,g)
        return Mult(self, other)
        
    def __sub__(self, other):
        # f-g redirects to Subtract(f,g)
        return Subtract(self, other)
            

class ZeroaryOp(Operation):
    # class for leaf operations, which do not depend on any other operations.
    # These include operations of type variable, integer constants, etc.
    def __init__(self):
        # self.children gives the list of children of self in the tree structure ; here empty
        self.children = []
        # self._Vars gives the set of all "variables", i.e. operations of type Var, in the formula
        # It is empty for leaf operations, except if self is itself of type Var
        self._Vars = {}


class UnaryOp(Operation):
    # class for unary operations, which depend on one child operation,
    # such as Exp(f), Sum(f), etc.
    
    def __init__(self, arg):
        # arg is another instance of Operation, it is the child operation of self
        self.children = [arg]
        # the set of variables in self is the set of variables in the child operation
        self._Vars = arg._Vars
        
    def Eval(self, out, table):
        # returns the C++ code string corresponding to the evaluation of the formula
        # - out is a c_variable in which the result of the evaluation is stored
        # - table is the list of c_variables corresponding to actual local variables
        # required for evaluation : each Var(ind,*,*) corresponds to table[ind]
        string = ""
        # Ealuation of the child operation
        if isinstance(self.children[0],Var):
            # if the child of the operation is a Var, we do not need to evaluate it,
            # we simply record the corresponding c_variable
            arg = table[self.children[0].ind]
        else:
            # otherwise, we need to evaluate the child operation.
            # We first create a new c_variable to store the result of the child operation.
            # This c_variable must have a unique name in the code, to avoid conflicts
            # when we will recursively evaluate nested operations.
            template_string_id = "out_" + self.children[0].string_id.lower()
            arg = new_c_variable(template_string_id,out.dtype)
            # Now we append into string the C++ code to declare the array
            string += f"{value(out.dtype)} {arg()}[{self.children[0].dim}];\n"
            # Now we evaluate the child operation and append the result into string
            string += self.children[0](arg, table)
        # Finally, evaluation of the operation itself
        string += self.Op(out, arg)
        return string
        
class BinaryOp(Operation):
    # class for unary operations, which depend on one child operation,
    # such as Mult(f,g), Subtract(f,g), etc.
    
    def __init__(self, arg0, arg1):
        # arg0 and arg1 are other instances of Operation, they are the child operations of self
        self.children = [arg0, arg1]
        # The variables in the current formula is the union of the variables in the child operations.
        # Note that this requires implementing properly __eq__ and __hash__ methods in Var class
        self._Vars = arg0._Vars.union(arg1._Vars)
        
    def Eval(self, out, table):
        # returns the C++ code string corresponding to the evaluation of the formula
        # - out is a c_variable in which the result of the evaluation is stored
        # - table is the list of c_variables corresponding to actual local variables
        # required for evaluation : each Var(ind,*,*) corresponds to table[ind]
        string = ""
        # Evaluation of the first child operation 
        if isinstance(self.children[0],Var):
            # if the child of the operation is a Var, we do not need to evaluate it,
            # we simply record the corresponding c_variable
            arg0 = table[self.children[0].ind]
        else:
            # otherwise, we need to evaluate the child operation.
            # We first create a new c_variable to store the result of the child operation.
            # This c_variable must have a unique name in the code, to avoid conflicts
            # when we will recursively evaluate nested operations.
            template_string_id = "out_" + self.children[0].string_id.lower()
            arg0 = new_c_variable(template_string_id,out.dtype)
            # Now we append into string the C++ code to declare the array
            string += f"{value(out.dtype)} {arg0()}[{self.children[0].dim}];\n"
            # Now we evaluate the child operation and append the result into string
            string += self.children[0](arg0, table)
        # Evalaution of the second child operation
        if isinstance(self.children[1],Var):
            arg1 = table[self.children[1].ind]
        else:
            template_string_id = "out_" + self.children[1].string_id.lower()
            arg1 = new_c_variable(template_string_id,out.dtype)
            string += f"{value(out.dtype)} {arg1()}[{self.children[1].dim}];\n"
            string += self.children[1](arg1, table)
        # Finally, evaluation of the operation itself
        string += self.Op(out, arg0, arg1)
        return string
        
        
class Var(ZeroaryOp):
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
        self._Vars = {self}
        
    # custom methods for printing the object
    def recursive_str(self, depth=0):
        return "Var({},{},{})".format(self.ind, self.dim, self.cat)
    def recursive_repr(self):
        return self.recursive_str()
        
    # custom __eq__ and __hash__ methods, required to handle properly the union of two sets of Var objects
    def __eq__(self, other):
        return self.ind == other.ind and self.dim == other.dim and self.cat == other.cat 
    def __hash__(self):
        return hash((self.ind,self.dim,self.cat))
    def Eval(self, out, table):
        return VectCopy(self.dim, out, table[self.ind], cast=False)    
    
    
    
class VectorizedScalarUnaryOp(UnaryOp):
    # class for unary operations that are vectorized scalar operations,
    # such as Exp(f), Cos(f), etc.
    
    @property
    def dim(self):
        # dim gives the output dimension of the operation, 
        # here it is the same as the output dimension of the child operation
        return self.children[0].dim
        
    def Op(self, out, arg):
        # Atomic evaluation of the operation : it consists in a simple
        # for loop around the call to the correponding scalar operation
        return VectApply(self.ScalarOp, self.dim, self.dim, out, arg)
        

class VectorizedScalarBinaryOp(BinaryOp):
    # class for binary operations that are vectorized or broadcasted scalar operations,
    # such as Mult(f), Subtract(f), etc.
    
    @property
    def dim(self):
        # dim gives the output dimension of the operation, 
        # here it is the max of the child operations (because we allow for broadcasting here)
        return max(self.children[0].dim, self.children[1].dim)
        
    def Op(self, out, arg0, arg1):
        # Atomic evaluation of the operation : it consists in a simple
        # for loop around the call to the correponding scalar operation
        return VectApply2(self.ScalarOp, self.dim, self.dim, self.dim, out, arg0, arg1)
        
class Exp(VectorizedScalarUnaryOp):
    # the exponential vectorized operation
    string_id = "Exp"
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = {keops_exp(arg)};\n"
    
class Minus(VectorizedScalarUnaryOp):
    # the "minus" vectorized operation
    string_id = "Minus"
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = -{arg()};\n"

class Square(VectorizedScalarUnaryOp):
    # the square vectorized operation
    string_id = "Square"
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = {arg()}*{arg()};\n"

class Sum(UnaryOp):
    # the summation operation
    string_id = "Sum"
    dim = 1
    def Op(self, out, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        string = f"*{out()} = {cast_to(value(out.dtype))}(0.0f);\n"
        string += VectApply(self.ScalarOp, 1, self.dim, out, arg)
        return string
    def ScalarOp(self,out,arg):
        return f"{out()} += {arg()};\n"

class Mult(VectorizedScalarBinaryOp):
    # the binary multiply operation
    string_id = "Mult"
    def ScalarOp(self,out,arg0,arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = {arg0()}*{arg1()};\n"

class Subtract(VectorizedScalarBinaryOp):
    # the binary subtract operation
    string_id = "Subtract"
    def ScalarOp(self,out,arg0,arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = {arg0()}-{arg1()};\n"




