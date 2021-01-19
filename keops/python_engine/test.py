
class tree:
    def __str__(self):
        return self.recursive_str()
    
    def recursive_str(self, depth=0):
        depth += 1
        string = self.string_id
        for child in self.children:
            string += "\n" + depth*4*" " + "{}".format(child.recursive_str(depth=depth))
        return string
       
    def __repr__(self):
        return self.recursive_repr()
    
    def recursive_repr(self):
        string = self.string_id + "("
        for child in self.children:
            string += "{},".format(child.recursive_str(depth=depth))
        string += ")"
        return string
       
class operation(tree):
    pass

class reduction(tree):
    def __init__(self, formula, tagIJ):
        self.formula = formula
        self.children = [formula]
        self.tagIJ = tagIJ
        
        
    
class zeroary(operation):
    string_id = "zeroary"
    def __init__(self):
        self.children = []
        self.vars = {}

class unary(operation):
    string_id = "unary"
    def __init__(self, arg):
        self.children = [arg]
        self.vars = arg.vars
        
class binary(operation):
    string_id = "binary"
    num_children = 1
    def __init__(self, arg0, arg1):
        self.children = [arg0, arg1]
        self.vars = arg0.vars.union(arg1.vars)
        
        
        
class Var(zeroary):
    string_id = "Var"
    def __init__(self, ind, dim, cat):
        super().__init__()
        self.ind = ind
        self.dim = dim
        self.cat = cat
        self.vars = {self}
    def recursive_str(self, depth=0):
        return "Var({},{},{})".format(self.ind, self.dim, self.cat)
    def recursive_repr(self):
        return self.recursive_str()
 

        
class ComplexSum(unary):
    string_id = "ComplexSum"
    dim = 2

class Real2Complex(unary):
    string_id = "Real2Complex"
    @property
    def dim():
        return 2*self.children[0].dim

class ComplexReal(unary):
    string_id = "ComplexReal"
    

class Add(binary):
    string_id = "Add"
    
class ComplexMult(binary):
    string_id = "ComplexMult"
    
    
def VectAssign(out, dim, val):
    return f"#pragma unroll for(int k=0; k<{dim}; k++) {out.string}[k] = cast_to<{out.dtype}>({val});"
        

class TemplatedCode:
    def __init__(self, dtypeacc, dtype):
        self.dtypeacc = dtypeacc
        self.dtype = dtype
        
class Sum_Reduction(reduction):
    string_id = "Sum_Reduction"
    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = formula.dim
    class InitializeReduction(TemplatedCode):
        def __call__(self, tmp):
            return VectAssign(self.dim)(tmp, "0.0f")
    class ReducePairScalar(TemplatedCode):
        def __call__(self, tmp, xi):
            return f"{tmp.string} += cast_to<{self.dtypeacc}>({xi.string});"
    class ReducePairShort(TemplatedCode):
        def __call__(self, tmp, xi, val):
            return VectApply(ReducePairScalar(self.dtypeacc,self.dtype), self.dim, self.dim)(tmp, xi)
        
    
    
class c_variable():
    def __init__(self, string, dtype):
        self.string = string
        self.dtype = dtype


    
    
formula = "Sum_Reduction(ComplexSum(Add(ComplexMult(Var(0,6,0), Var(1,6,1)), Real2Complex(ComplexReal(Var(0,6,0))))),1)"

f = eval(formula)
print(f)
print(f.formula.vars)
print(list(f.formula.vars)[0])

dtypeacc = "float"
dtype = "float"
tmp = c_variable("tmp", dtype)

print(f.InitializeReduction(dtypeacc, dtype)(tmp))
