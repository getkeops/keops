from math import prod

# utils

def check_get_unique_attr(objects, attr):
    values = set(getattr(obj,attr) for obj in objects)
    if len(values)!=1:
        raise ValueError(f"incompatible {attr}")
    return values.pop()

class Node:

    def inner_string(self, str_args):
        return str_args

    def __repr__(self, *str_args):
        str_inner = ",".join(str_elem for str_elem in self.inner_string(str_args))
        return f"{self.string_id}({str_inner})"

class Tree:
    def __init__(self, node=Node(), children=()):
        self.node = node
        self.children = children
    def __repr__(self):
        str_args = [x.__repr__() for x in self.children]
        return self.node.__repr__(*str_args)
    def collect(self, fun_test=lambda x:True):
        res = [self] if fun_test(self) else []
        for child in self.children:
            res += child.collect(fun_test)
        return res



# generic symbolic tensors
class GenericSymbolicTensor(Tree):
    _shape = None

    @property
    def formula(self):
        return self.__repr__() # TODO formula and repr methods should return different strings...

    @property
    def shape(self):
        return self._shape
    
    @property
    def Vars(self):
        return self.collect(lambda x : isinstance(x,GenericVariable))

    def assign_variables_indices(self):
        vars = self.Vars
        ind = 0
        for var in vars:
            var.ind = ind
            ind += 1
        return vars
    
    def dense(self):
        if not isinstance(self.node, ReductionOp):
            raise ValueError("not implemented yet")
        reduction_op = self.node.string_id
        reduction_axis = self.node.axis
        inner_formula = self.children[0].__repr__()
        vars = self.assign_variables_indices()
        keops_fun = Genred(inner_formula,[],reduction_op=reduction_op,axis=reduction_axis)
        input()
        return keops_fun(*(var.tensor for var in vars))

    def exp(self):
        return exp(self)

    def __sub__(self,other):
        return SubOp()(self,other)

    #def __getitem__(self, key):
    #    if isinstance(key, slice):



class GenericVariable(GenericSymbolicTensor):
    def __init__(self, tensor, nbatchdims=0):
        shape = tensor.shape
        assert(len(shape)>=nbatchdims+2)
        assert(shape[nbatchdims]==1 or shape[nbatchdims+1]==1)
        self.tensor = tensor
        self._shape = shape
        self.ind = id(tensor)
        self.dim = prod(shape[nbatchdims+2:])
        if shape[nbatchdims]==1:
            if shape[nbatchdims+1]==1:
                self.cat = 2
            else:
                self.cat = 1
        else: # here shape[nbatchdims+1]==1
            self.cat = 0
        super().__init__()
    
    def __repr__(self):
        return f"Var({self.ind},{self.dim},{self.cat})"

    #def __getitem__(self, key):
    #    self.SymbolicTensorConstructor(self.tensor[key], nbatchdims=self.nbatchdims)
 

# rules for checking and infering shapes
class BroadcastShapes:
    @staticmethod
    def get_shape(args, params):
        shapes = [arg.shape for arg in args]
        ndims = set(len(shape) for shape in shapes)
        if len(ndims)!=1:
            raise ValueError("incompatible shapes : different number of dimensions")
        ndims = ndims.pop()
        shapeout = []
        for k in range(ndims):
            dims = set(shape[k] for shape in shapes)
            if len(dims)==2 and min(dims)==1:
                dimout = max(dims)
            elif len(dims)==1:
                dimout = dims.pop()
            else:
                raise ValueError(f"incompatible shapes : k={k}, dims={dims}")
            shapeout.append(dimout)
        return tuple(shapeout)

class ReductionShape:
    @staticmethod
    def get_shape(args, params):
        arg, = args  # there should be only one argument by default
        axis, = params # the only parameter should be the axis of reduction by default
        shapeout = list(arg.shape)
        shapeout[axis] = 1
        return tuple(shapeout)

# operations :

class Op(Node):

    def __init__(self, params=()):
        self.params = params

    def inner_string(self, str_args):
        str_params = [param.__repr__() for param in self.params]
        return (*str_args,*str_params)

    def __call__(self, *args):
        backend = check_get_unique_attr(args,"SymbolicTensor_backend")
        res = backend.SymbolicTensorConstructor(node=self, children=args)
        res._shape = self.get_shape(args, self.params)
        return res



class ScalarOp(Op, BroadcastShapes):
    pass

class ExpOp(ScalarOp):
    string_id = "Exp"

def exp(x):
    return ExpOp()(x)
    
class SubOp(ScalarOp):
    string_id = "Subtract"


class ReductionOp(Op, ReductionShape):
    def __init__(self, axis):
        self.params = (axis,)
    @property
    def axis(self):
        return self.params[0]

class SumReductionOp(ReductionOp):
    string_id = "Sum"

def sum_reduction(x, axis=None):
    return SumReductionOp(axis)(x)

# pytorch backend

from pykeops.torch import Genred

class PytorchSymbolicTensor(GenericSymbolicTensor):
    @property
    def SymbolicTensor_backend(self):
        return PytorchSymbolicTensor
    @staticmethod
    def SymbolicTensorConstructor(node=Op(), children=()):
        return PytorchSymbolicTensor(node=node, children=children)

    
    

class Variable(GenericVariable, PytorchSymbolicTensor):
    def __init__(self, tensor, nbatchdims):
        super().__init__(tensor, nbatchdims)

def SymbolicTensor(tensor, nbatchdims=0):
    return Variable(tensor, nbatchdims)




# example

import torch

x = torch.rand(400,1,3)
y = torch.rand(1,300,3)
xi = SymbolicTensor(x)
yj = SymbolicTensor(y)
K = exp(xi-yj)
print(K)
print(K.shape)
K.assign_variables_indices()
print(K)

res = sum_reduction(K, axis=1)

res.dense()



    
