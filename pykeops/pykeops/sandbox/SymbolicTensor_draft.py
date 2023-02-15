###############
###  utils
###############

from math import prod


def check_get_unique_attr(objects, attr):
    # given a list of objects, make sure the attribute attr
    # is the same for each object, and return this common attribute
    values = set(getattr(obj, attr) for obj in objects)
    if len(values) != 1:
        raise ValueError(f"incompatible {attr}")
    return values.pop()


class Node:
    # default class for representing a node of a tree structure
    node_id = "Node"  # can be specialized

    def __repr__(self, *str_args):
        if hasattr(self, "params"):
            str_args = list(str_args) + [str(param) for param in self.params]
        str_inner = ",".join(str_elem for str_elem in str_args)
        return f"{self.node_id}({str_inner})"


class Tree:
    # implements tree structures
    # a Tree object has a node attribute, and a list of children

    def __init__(self, node=Node(), children=()):
        self.node = node
        self.children = children

    def recursive_str(self, method):
        # common method for recursively building a string from a tree structure
        str_args = [x.recursive_str(method) for x in self.children]
        return getattr(self.node, method)(*str_args)

    def __repr__(self):
        return self.recursive_str("__repr__")

    def collect(self, fun_test=lambda x: True):
        # build the list of all subtrees of a tree that satisfy a given condition
        # given by the function fun_test
        # example : T.collect(fun_test=lambda x : isinstance(x,A)) gives the list of subtrees
        # of the tree T that are instances of class A
        res = [self] if fun_test(self) else []
        for child in self.children:
            res += child.collect(fun_test)
        return res


#####################################
###  generic symbolic tensors
#####################################


class GenericSymbolicTensor(Tree):

    # this is the main class for symbolic tensors

    _shape = None

    def keops_formula(self):
        # self.keops_formula() returns a string that corrsponds
        # to the object to be instanciated with keopscore
        return self.recursive_str("keops_formula")

    @property
    def shape(self):
        # this is the shape of the symbolic tensor. It must be identical
        # to the shape of the equivalent tensor if we had performed
        # the same sequence of operations over actual tensors.
        return self._shape

    def __str__(self):
        self.assign_variables_indices()
        return f"{type(self).__name__} with shape {self.shape} and formula {self.keops_formula()}"

    @property
    def Vars(self):
        # return the list of variables of the symbolic tensor
        return self.collect(lambda x: isinstance(x, GenericVariable))

    def assign_variables_indices(self):
        # assigns a unique index to each variable in the symbolic tensor,
        # from 0 to n-1 where n is the number of variables.
        # The index of a variable correponds to the position of its corresponding
        # input tensor in the list of arguments of the final reduction call.
        vars = self.Vars
        ind = 0
        for var in vars:
            var.ind = ind
            ind += 1
        return vars

    def dense(self):
        # returns the actual tensor corresponding to self by performing
        # the sequence of operations encoded in the symbolic tensor.
        # This is done by coalling Genred
        if not isinstance(self.node, ReductionOp):
            raise ValueError("not implemented yet")
        reduction_op = self.node.keops_string_id
        reduction_axis = self.node.axis
        vars = self.assign_variables_indices()
        inner_formula = self.children[0].keops_formula()
        keops_fun = Genred(
            inner_formula, [], reduction_op=reduction_op, axis=reduction_axis
        )
        return keops_fun(*(var.tensor for var in vars)).reshape(self.shape)

    # below are aliases for operations on symbolic tensors

    def exp(self):
        return exp(self)

    def __neg__(self):
        return MinusOp()(self)

    def __sub__(self, other):
        return SubOp()(self, other)

    def __mul__(self, other):
        return MultOp()(self, other)

    def __pow__(self, order):
        if order == 2:
            return SquareOp()(self)
        else:
            raise ValueError("not implemented")

    def sum(self, axis, keepdim=False):
        return sum(self, axis, keepdim)

    # def __getitem__(self, key):
    #    if isinstance(key, slice):


class GenericVariable(GenericSymbolicTensor):
    # the generic variable class
    def __init__(self, tensor, nbatchdims=0):
        shape = tensor.shape
        assert len(shape) >= nbatchdims + 2
        assert shape[nbatchdims] == 1 or shape[nbatchdims + 1] == 1
        self.tensor = tensor
        self.nbatchdims = nbatchdims
        self._shape = shape
        self.ind = id(tensor)
        self.dim = prod(shape[nbatchdims + 2 :])
        if shape[nbatchdims] == 1:
            if shape[nbatchdims + 1] == 1:
                self.cat = 2
            else:
                self.cat = 1
        else:  # here shape[nbatchdims+1]==1
            self.cat = 0
        super().__init__(node=self)

    def keops_formula(self):
        return f"Var({self.ind},{self.dim},{self.cat})"

    def __repr__(self):
        return self.keops_formula()

    # def __getitem__(self, key):
    #    self.SymbolicTensorConstructor(self.tensor[key], nbatchdims=self.nbatchdims)


################################################
###  rules for checking and infering shapes
################################################
class BroadcastShapes:
    @staticmethod
    def get_shape(args, params=[]):
        # implements the broadcasting rule for shapes :
        # given a list of arguments x,y,z,... representing tensors
        # we check that their shapes are compatible for broadcasting
        # and return the output broadcasted shape.
        # for example if x.shape=(2,3,1) and y.shape=(2,1,4), then
        # get_shape([x,y]) will return (2,3,4)
        # N.B. input params is unused here.
        shapes = [arg.shape for arg in args]
        ndims = set(len(shape) for shape in shapes)
        if len(ndims) != 1:
            raise ValueError("incompatible shapes : different number of dimensions")
        ndims = ndims.pop()
        shapeout = []
        for k in range(ndims):
            dims = set(shape[k] for shape in shapes)
            if len(dims) == 2 and min(dims) == 1:
                dimout = max(dims)
            elif len(dims) == 1:
                dimout = dims.pop()
            else:
                raise ValueError(f"incompatible shapes : k={k}, dims={dims}")
            shapeout.append(dimout)
        return tuple(shapeout)


class ReductionShape:
    @staticmethod
    def get_shape(args, params):
        # implements the reduction rule for shapes.
        (arg,) = args  # there should be only one argument by default
        axis, keepdim = params  # TODO change this...
        shapeout = list(arg.shape)
        if keepdim:
            shapeout[axis] = 1
        else:
            shapeout.pop(axis)
        return tuple(shapeout)


#####################################################
###  classes for operations over symbolic tensors
#####################################################
class Op(Node):

    # base class for operations

    def __init__(self, params=[]):
        self.params = params
        self.keops_params = params

    @property
    def node_id(self):
        return self.keops_string_id

    def keops_formula(self, *str_args):
        str_params = [param.__repr__() for param in self.keops_params]
        str_inner = ",".join(str_elem for str_elem in (*str_args, *str_params))
        return f"{self.keops_string_id}({str_inner})"

    def __call__(self, *args):
        # *args is the sequence of arguments (children symbolic tensors) of the operation.
        # this returns the resulting symbolic tensor.
        backend = check_get_unique_attr(args, "SymbolicTensor_backend")
        res = backend.SymbolicTensorConstructor(node=self, children=args)
        res.nbatchdims = check_get_unique_attr(args, "nbatchdims")
        res._shape = self.get_shape(args, self.params)
        return res


class ScalarOp(Op, BroadcastShapes):
    # class for all scalar broadcasted operations
    pass


class ExpOp(ScalarOp):
    keops_string_id = "Exp"


def exp(x):
    return ExpOp()(x)


class MinusOp(ScalarOp):
    keops_string_id = "Minus"


class SubOp(ScalarOp):
    keops_string_id = "Subtract"


class SquareOp(ScalarOp):
    keops_string_id = "Square"


class MultOp(ScalarOp):
    keops_string_id = "Mult"


class SumOp(Op, ReductionShape):
    keops_string_id = "Sum"

    def __init__(self, axis, keepdim=False):  # N.B TODO axis is unused currently...
        self.params = [axis, keepdim]
        self.keops_params = []

    @property
    def axis(self):
        return self.params[0]

    @property
    def keepdim(self):
        return self.params[1]


class ReductionOp(Op, ReductionShape):
    # base class for reduction operations
    def __init__(self, axis, keepdim=False):
        self.params = [axis, keepdim]
        self.keops_params = [axis]

    @property
    def axis(self):
        return self.params[0]

    @property
    def keepdim(self):
        return self.params[1]


class SumReductionOp(ReductionOp):
    keops_string_id = "Sum"


def sum(x, axis=None, keepdim=False):
    if axis < x.nbatchdims:
        raise ValueError("not implemented")
    elif axis in [x.nbatchdims, x.nbatchdims + 1]:
        return SumReductionOp(axis - x.nbatchdims, keepdim)(x)
    else:
        if len(x.shape) != x.nbatchdims + 3:
            raise ValueError("not implemented")
        return SumOp(axis - x.nbatchdims, keepdim)(x)


#####################################################
###  specialized classes for PyTorch backend
#####################################################

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


#####################################################
###  decorator
#####################################################


def jit(fun):
    def newfun(*args, **kwargs):
        newargs = [SymbolicTensor(arg) for arg in args]
        res = fun(*newargs, **kwargs)
        return res.dense()

    return newfun


#####################################################
###  example of use
#####################################################

import torch

M, N, D = 4, 3, 2
x = torch.rand(M, 1, D)
y = torch.rand(1, N, D)
b = torch.rand(1, N)


def gauss_kernel(x, y, b):
    D2 = ((x - y) ** 2).sum(axis=2)
    K = (-D2).exp()
    f = K * b
    return f.sum(axis=1)


out1 = gauss_kernel(x, y, b)
print(out1.shape)


@jit
def gauss_kernel(x, y, b):
    D2 = ((x - y) ** 2).sum(axis=2)
    K = (-D2).exp()
    f = K * b
    return f.sum(axis=1)


out2 = gauss_kernel(x, y, b)
print(out2.shape)

print(torch.norm(out1 - out2))
