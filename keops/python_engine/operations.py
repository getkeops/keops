from tree import tree
from utils import new_c_variable, VectApply, VectCopy, VectApply2, value, keops_exp, cast_to

class Operation(tree):
    
    # Base class for all keops building block operations in a formula
    
    def __init__(self, *args):
        # *args are other instances of Operation, they are the child operations of self
        self.children = args
        # The variables in the current formula is the union of the variables in the child operations.
        # Note that this requires implementing properly __eq__ and __hash__ methods in Var class
        self._Vars = set.union(*(arg._Vars for arg in args)) if len(args)>0 else set()
        
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
                # We first create a new c_variable to store the result of the child operation.
                # This c_variable must have a unique name in the code, to avoid conflicts
                # when we will recursively evaluate nested operations.
                template_string_id = "out_" + child.string_id.lower()
                arg = new_c_variable(template_string_id,out.dtype)
                # Now we append into string the C++ code to declare the array
                string += f"{value(out.dtype)} {arg()}[{child.dim}];\n"
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
        
    def __add__(self, other):
        # f+g redirects to Add(f,g)
        return Add(self, other)
            
    def __sub__(self, other):
        # f-g redirects to Subtract(f,g)
        return Subtract(self, other)
            
    def __neg__(self):
        # -f redirects to Minus(f)
        return Minus(self)
            

        
        
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
        
    def Op(self, out, table):
        return VectCopy(self.dim, out, table[self.ind], cast=False)    
    
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
        
    # custom methods for printing the object
    def recursive_str(self, depth=0):
        return "Zero({})".format(self.dim)
    def recursive_repr(self):
        return self.recursive_str()
        
    def Op(self, out, table):
        zero = c_variable("0.0f","float")
        return VectAssign(self.dim, out, zero)

    def DiffT(self,v,gradin):
        return Zero(v.dim)
    
class IntConstant(Operation):
    # constant integer "operation"
    string_id = "IntConstant"
    
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.dim = 1
        
    # custom methods for printing the object
    def recursive_str(self, depth=0):
        return "IntConstant({})".format(self.value)
    def recursive_repr(self):
        return self.recursive_str()
        
    def Op(self, out, table):
        return f"*{out()} = {cast_to(out.dtype)}((float){self.value});\n"

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
        return VectApply(self.ScalarOp, self.dim, self.dim, out, arg)
        

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
        return VectApply2(self.ScalarOp, self.dim, self.dim, self.dim, out, arg0, arg1)
        
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
    
class Minus(VectorizedScalarUnaryOp):
    # the "minus" vectorized operation
    string_id = "Minus"
    def __new__(self, arg):
        if isinstance(arg,Zero):
            return arg
        else:
            return super().__new__(self)
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = -{arg()};\n"
    def DiffT(self,v,gradin):
        f = self.children[0]
        return -f.DiffT(v,gradin)
        
class Square(VectorizedScalarUnaryOp):
    # the square vectorized operation
    string_id = "Square"
    def __new__(self, arg):
        if isinstance(arg,Zero):
            return arg
        else:
            return super().__new__(self)
    def ScalarOp(self,out,arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = {arg()}*{arg()};\n"
    def DiffT(self,v,gradin):
        # [\partial_V (F)**2].gradin = F * [\partial_V F].gradin
        f = self.children[0]
        return IntConstant(2) * f.DiffT(v, f*gradin);
        
class Sum(Operation):
    # the summation operation
    string_id = "Sum"
    def __new__(self, arg):
        if isinstance(arg,Zero):
            return Zero(1)
        else:
            return super().__new__(self)
    dim = 1
    def Op(self, out, table, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        string = f"*{out()} = {cast_to(value(out.dtype))}(0.0f);\n"
        string += VectApply(self.ScalarOp, 1, self.dim, out, arg)
        return string
    def ScalarOp(self,out,arg):
        return f"{out()} += {arg()};\n"
    def DiffT(self,v,gradin):
        f = self.children[0]
        return f.DiffT(v,SumT(gradin,f.dim))

class SumT(Operation):
    # the adjoint of the summation operation
    string_id = "SumT"
    def __new__(self, arg, dim):
        if isinstance(arg,Zero):
            return Zero(dim)
        else:
            return super().__new__(self)
    def __init__(self, arg, dim):
        super().__init__(arg)
        self.dim = dim
    def Op(self, out, table, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        value_arg = c_variable(f"*{arg()}",value(arg.dtype))
        return VectAssign(self.dim,out,value_arg)
    def DiffT(self,v,gradin):
        f = self.children[0]
        return f.DiffT(v,Sum(gradin))

class Mult(VectorizedScalarBinaryOp):
    # the binary multiply operation
    string_id = "Mult"
    def __new__(self, arg0, arg1):
        if isinstance(arg0,Zero):
            return arg0
        elif isinstance(arg1,Zero):
            return arg1
        else:
            return super().__new__(self)
    def ScalarOp(self,out,arg0,arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = {arg0()}*{arg1()};\n"
    #  \diff_V (A*B) = (\diff_V A) * B + A * (\diff_V B)    
    def DiffT(self,v,gradin):
        fa, fb = self.children
        return fa.DiffT(v,fb*gradin) + fb.DiffT(v,fa*gradin)

class Add(VectorizedScalarBinaryOp):
    # the binary addition operation
    string_id = "Add"
    def __new__(self, arg0, arg1):
        if isinstance(arg0,Zero):
            return arg1
        elif isinstance(arg1,Zero):
            return arg0
        else:
            return super().__new__(self)
    def ScalarOp(self,out,arg0,arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = {arg0()}+{arg1()};\n"
    def DiffT(self,v,gradin):
        fa, fb = self.children
        return fa.DiffT(v,gradin) + fb.DiffT(v,gradin)

class Subtract(VectorizedScalarBinaryOp):
    # the binary subtract operation
    string_id = "Subtract"
    def __new__(self, arg0, arg1):
        if isinstance(arg0,Zero):
            return -arg1
        elif isinstance(arg1,Zero):
            return arg0
        else:
            return super().__new__(self)
    def ScalarOp(self,out,arg0,arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"{out()} = {arg0()}-{arg1()};\n"
    def DiffT(self,v,gradin):
        fa, fb = self.children
        return fa.DiffT(v,gradin) - fb.DiffT(v,gradin)




