from tree import tree
from utils import new_c_variable, VectApply, VectCopy, VectApply2, value, keops_exp

class Operation(tree):
    def __mul__(self, other):
        return Mult(self, other)
    def __sub__(self, other):
        return Subtract(self, other)
    def Vars(self, cat="all"):
        if cat=="all":
            return self.Vars_
        else:
            res = []
            for v in self._Vars:
                if v.cat == cat:
                    res.append(v)
            return res
            

class ZeroaryOp(Operation):
    string_id = "ZeroaryOp"
    def __init__(self):
        self.children = []
        self._Vars = {}

class UnaryOp(Operation):
    string_id = "UnaryOp"
    def __init__(self, arg):
        self.children = [arg]
        self._Vars = arg._Vars
    def __call__(self, out, table):
        string = ""
        if isinstance(self.children[0],Var):
            arg = table[self.children[0].ind]
        else:
            arg = new_c_variable("arg",out.dtype)
            string += f"{value(out.dtype)} {arg()}[{self.children[0].dim}];\n"
            string += self.children[0](arg, table)
        string += self.Op(out, arg)
        return string
        
class BinaryOp(Operation):
    string_id = "BinaryOp"
    num_children = 1
    def __init__(self, arg0, arg1):
        self.children = [arg0, arg1]
        self._Vars = arg0._Vars.union(arg1._Vars)
    def __call__(self, out, table):
        string = ""
        if isinstance(self.children[0],Var):
            arg0 = table[self.children[0].ind]
        else:
            arg0 = new_c_variable("arg0",out.dtype)
            string += f"{value(out.dtype)} {arg0()}[{self.children[0].dim}];\n"
            string += self.children[0](arg0, table)
        if isinstance(self.children[1],Var):
            arg1 = table[self.children[1].ind]
        else:
            arg1 = new_c_variable("arg1",out.dtype)
            string += f"{value(out.dtype)} {arg1()}[{self.children[1].dim}];\n"
            string += self.children[1](arg1, table)
        string += self.Op(out, arg0, arg1)
        return string
        
class Var(ZeroaryOp):
    string_id = "Var"
    def __init__(self, ind, dim, cat):
        super().__init__()
        self.ind = ind
        self.dim = dim
        self.cat = cat
        self._Vars = {self}
    def recursive_str(self, depth=0):
        return "Var({},{},{})".format(self.ind, self.dim, self.cat)
    def recursive_repr(self):
        return self.recursive_str()
    def __eq__(self, other):
        return self.ind == other.ind and self.dim == other.dim and self.cat == other.cat 
    def __hash__(self):
        return hash((self.ind,self.dim,self.cat))
    def __call__(self, out, table):
        return VectCopy(self.dim, out, table[self.ind], cast=False)    
    
class VectorizedScalarUnaryOp(UnaryOp):
    string_id = "VectorizedScalarUnaryOp"
    @property
    def dim(self):
        return self.children[0].dim
    def Op(self, out, arg):
        return VectApply(self.ScalarOp, self.dim, self.dim, out, arg)
        

class VectorizedScalarBinaryOp(BinaryOp):
    string_id = "VectorizedScalarBinaryOp"
    @property
    def dim(self):
        return max(self.children[0].dim, self.children[1].dim)
    def Op(self, out, arg0, arg1):
        return VectApply2(self.ScalarOp, self.dim, self.dim, self.dim, out, arg0, arg1)

class Exp(VectorizedScalarUnaryOp):
    string_id = "Exp"
    def ScalarOp(self,out,arg):
        return f"{out()} = {keops_exp(arg)};\n"
    
class Minus(VectorizedScalarUnaryOp):
    string_id = "Minus"
    def ScalarOp(self,out,arg):
        return f"{out()} = -{arg()};\n"

class Square(VectorizedScalarUnaryOp):
    string_id = "Square"
    def ScalarOp(self,out,arg):
        return f"{out()} = {arg()}*{arg()};\n"

class Sum(UnaryOp):
    string_id = "Sum"
    dim = 1
    def Op(self, out, arg):
        string = f"*{out()} = cast_to<{value(out.dtype)}>(0.0f);\n"
        return VectApply(self.ScalarOp, 1, self.dim, out, arg)
    def ScalarOp(self,out,arg):
        return f"{out()} += {arg()};\n"

class Mult(VectorizedScalarBinaryOp):
    string_id = "Mult"
    def ScalarOp(self,out,arg0,arg1):
        return f"{out()} = {arg0()}*{arg1()};\n"

class Subtract(VectorizedScalarBinaryOp):
    string_id = "Subtract"
    def ScalarOp(self,out,arg0,arg1):
        return f"{out()} = {arg0()}-{arg1()};\n"