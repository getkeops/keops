import torch

class ReproducibleMeta(type):
    def __new__(cls, name, bases, body):
        if name != 'Reproducible' and '__init__' in body:
            raise TypeError(f"class {name} derives from Reproducible, so it should not define __init__ method. Define ___init___ instead.")
        return super().__new__(cls, name, bases, body)
        
class Reproducible(metaclass=ReproducibleMeta):
    
    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs
        self.___init___(*args, **kwargs)
    
    def ___init___(*args, **kwargs):pass
    
    def __repr__(self):
        str_args = [str(arg) for arg in self.__args]
        str_args += [f"{key}={self.kwargs[key]}" for key in self.__kwargs]
        str_inner = ",".join(str_elem for str_elem in str_args)
        return f"{self.__class__.__name__}({str_inner})" 
    
    def __eq__(self, other):
        test_type = type(self)==type(other)
        test_args = self.__args==other.__args
        test_kwargs = self.__kwargs==other.__kwargs
        return test_type and test_args and test_kwargs

class Tree(Reproducible):
    # implements tree structures
    # a Tree object has a node attribute, and a list of children

    def ___init___(self, children=(), params={}):
        assert(all(isinstance(child,Tree) for child in children))
        assert(isinstance(params,dict))
        self.children = children
        self.params = params

    def collect(self, fun_test=lambda x: True):
        # build the list of all subtrees of a tree that satisfy a given condition
        # given by the function fun_test
        # example : T.collect(fun_test=lambda x : isinstance(x,A)) gives the list of subtrees
        # of the tree T that are instances of class A
        res = [self] if fun_test(self) else []
        for child in self.children:
            res += child.collect(fun_test)
        return res
    
    def print_tree(self, indent=0):
        string = f"{' '*indent}{self.__class__.__name__}"
        if len(self.params)>0:
            string += f" {self.params}"
        if len(self.children)>0:
            string += " with children:"
        print(string)
        for child in self.children:
            child.print_tree(indent+2)
    
    def recursive(self, method):
            # common method for recursively applying a method for a tree object
            str_args = [x.recursive(method) for x in self.children]
            return getattr(self, method)(*str_args)
            

class Op(Tree):
    def ___init___(self,*args, params={}):
        assert(isinstance(params,dict))
        super().___init___(children=args, params=params)
        
class Mult(Op):
    pass
        
class Pow(Op):
    def ___init___(self,x,n):
        assert(isinstance(n,int))
        super().___init___(x, params={"n":n})

class GenericVar(Op):
    tensor_class = object
    def ___init___(self, x):
        assert(isinstance(x, self.tensor_class))
        self.tensor = x
        super().___init___(params={"tensor":x})
    def __str__(self):
        return f"Var(<{id(self.tensor)}>)"

class Var(GenericVar):
    tensor_class = torch.Tensor
    
x = Var(torch.rand(2,3))
y = Var(torch.rand(2,3))

f = Mult(Mult(x,y),Pow(x,3))

print(f)



