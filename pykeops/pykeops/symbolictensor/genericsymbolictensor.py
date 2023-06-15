#####################################
###  generic symbolic tensors
#####################################

import pykeops
from pykeops.symbolictensor.utils import Tree
from math import prod


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

    @property
    def inner_shape(self):
        # this is the inner shape of the symbolic tensor, corresponding
        # to the shape of the tensor inside the reduction formula
        # (i.e. the shape without the batch dimensions and without the i and j axes)
        return self._shape[self.nbatchdims + 2 :]

    @property
    def outer_shape(self):
        # this is the outer shape of the symbolic tensor, corresponding
        # to the batch dimensions and the i and j axes
        return self._shape[: self.nbatchdims + 2]

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
        # This is done by calling Genred
        from pykeops.symbolictensor.operations import ReductionOp

        if not isinstance(self.node, ReductionOp):
            raise ValueError("not implemented yet")
        reduction_op = self.node.keops_string_id
        reduction_axis = self.node.axis
        vars = self.assign_variables_indices()
        inner_formula = self.children[0].keops_formula()
        keops_fun = self.Genred(
            inner_formula, [], reduction_op=reduction_op, axis=reduction_axis
        )
        keops_inputs = (
            var.tensor.view(var.outer_shape + (prod(var.inner_shape),)) for var in vars
        )
        return keops_fun(*keops_inputs).reshape(self.shape)

    # below are aliases for operations on symbolic tensors

    def exp(self):
        return pykeops.symbolictensor.operations.exp(self)

    def __neg__(self):
        return pykeops.symbolictensor.operations.MinusOp()(self)

    def __sub__(self, other):
        return pykeops.symbolictensor.operations.SubOp()(self, other)

    def __add__(self, other):
        return pykeops.symbolictensor.operations.AddOp()(self, other)

    def __mul__(self, other):
        return pykeops.symbolictensor.operations.MultOp()(self, other)

    def __truediv__(self, other):
        return pykeops.symbolictensor.operations.DivideOp()(self, other)

    def __pow__(self, order):
        if order == 2:
            return pykeops.symbolictensor.operations.SquareOp()(self)
        else:
            raise ValueError("not implemented")

    def sum(self, axis, keepdim=False):
        return pykeops.symbolictensor.operations.sum(self, axis, keepdim)

    # def __getitem__(self, key):
    #    if isinstance(key, slice):


class GenericVariable(GenericSymbolicTensor):
    # the generic variable class
    def __init__(self, tensor, nbatchdims=0):
        shape = tuple(tensor.shape)
        assert len(shape) >= nbatchdims + 2
        assert shape[nbatchdims] == 1 or shape[nbatchdims + 1] == 1
        self.tensor = tensor
        self.nbatchdims = nbatchdims
        self._shape = shape
        self.ind = id(tensor)
        if shape[nbatchdims] == 1:
            if shape[nbatchdims + 1] == 1:
                self.cat = 2
            else:
                self.cat = 1
        else:  # here shape[nbatchdims+1]==1
            self.cat = 0
        super().__init__(node=self)

    def keops_formula(self):
        return f"Var({self.ind},{prod(self.inner_shape)},{self.cat})"

    def __repr__(self):
        return self.keops_formula()

    # def __getitem__(self, key):
    #    self.SymbolicTensorConstructor(self.tensor[key], nbatchdims=self.nbatchdims)
