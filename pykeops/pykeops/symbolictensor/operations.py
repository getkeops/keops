#####################################################
###  classes for operations over symbolic tensors
#####################################################

import pykeops
from pykeops.symbolictensor.utils import Node, check_get_unique_attr
from pykeops.symbolictensor.shapes import BroadcastShapes, ReductionShape


class Op(Node):
    # base class for operations

    def __init__(self, *keops_params):
        self.params = {}
        self.keops_params = keops_params

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
        res._shape = self.get_shape(*args, **self.params)
        return res


class ScalarOp(Op, BroadcastShapes):
    # class for all scalar broadcasted operations
    def __call__(self, *args):
        res = super().__call__(*args)
        if self.test_non_trivial_inner_broadcast(args):
            self.keops_params = [[arg.inner_shape for arg in (res, *args)]]
        if pykeops.symbolictensor.debug_mode:
            print("\nIn __call__ method of class ScalarOp")
            print("  res.keops_formula()=", res.keops_formula())
        return res
    
class InnerReductionOp(Op):
    # class for all inner reduction operations like Sum, Max, etc.
    def __call__(self, arg, axis=None, keepdim=False):
        res = super().__call__(arg,params=(axis,keepdim))
        if pykeops.symbolictensor.debug_mode:
            print("\nIn __call__ method of class ScalarOp")
            print("  res.keops_formula()=", res.keops_formula())
        return res


class ExpOp(ScalarOp):
    keops_string_id = "Exp"


def exp(x):
    return ExpOp()(x)


class AddOp(ScalarOp):
    keops_string_id = "Add"


class MinusOp(ScalarOp):
    keops_string_id = "Minus"


class SubOp(ScalarOp):
    keops_string_id = "Subtract"


class SquareOp(ScalarOp):
    keops_string_id = "Square"


class MultOp(ScalarOp):
    keops_string_id = "Mult"


class DivideOp(ScalarOp):
    keops_string_id = "Divide"


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
    def __init__(self, axis=None, keepdim=False):
        self.params = {"axis":axis, "keepdim":keepdim}
        self.keops_params = [axis]

    @property
    def axis(self):
        return self.params["axis"]

    @property
    def keepdim(self):
        return self.params["keepdim"]


class SumReductionOp(ReductionOp):
    keops_string_id = "Sum"


def sum(x, axis=None, keepdim=False):
    if axis < x.nbatchdims:
        raise ValueError("not implemented")
    elif axis in [x.nbatchdims, x.nbatchdims + 1]:
        return SumReductionOp(axis=axis - x.nbatchdims, keepdim=keepdim)(x)
    else:
        if len(x.shape) != x.nbatchdims + 3:
            raise ValueError("not implemented")
        return SumOp(axis - x.nbatchdims, keepdim)(x)
