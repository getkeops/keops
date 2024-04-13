from keopscore.utils.code_gen_utils import VectApply, c_variable
from keopscore.formulas.Operation import Operation
from keopscore.utils.misc_utils import KeOps_Error


class VectorizedScalarOp(Operation):
    # class for operations that are vectorized or broadcasted
    # scalar operations,
    # such as Exp(f), Cos(f), Mult(f,g), Subtract(f,g), etc.

    def __new__(cls, *args, params=(), allow_fuse=True):
        obj = super(VectorizedScalarOp, cls).__new__(cls) 
        obj.__init__(*args, params=params)
        if allow_fuse:
            for ind, arg in enumerate(args):
                if isinstance(arg, VectorizedScalarOp):
                    fused_args = args[:ind] + arg.children + args[ind+1:]
                    return FusedVectorizedScalarOp.__new__(FusedVectorizedScalarOp, *fused_args, params=(obj, ind), allow_fuse=False)
        return obj

    def __init__(self, *args, params=()):
        dims = set(arg.dim for arg in args)
        if len(dims) > 2 or (len(dims) == 2 and min(dims) != 1):
            KeOps_Error("dimensions are not compatible for VectorizedScalarOp")
        super().__init__(*args, params=params)

    @property
    def dim(self):
        # dim gives the output dimension of the operation,
        # here it is the same as the output dimension of the child operation
        return max(child.dim for child in self.children)

    def Op(self, out, table, *args):
        # Atomic evaluation of the operation : it consists in a simple
        # for loop around the call to the correponding scalar operation
        return VectApply(self.ScalarOp, out, *args)

    def ScalarOp(self, out, *args):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        res = out.assign(type(self).ScalarOpFun(*args, *self.params))
        return res

    def DiffT(self, v, gradin):
        derivatives = self.Derivative(*self.children, *self.params)
        if len(self.children) == 1:
            derivatives = (derivatives,)
        return sum(
            (f.DiffT(v, gradin) * df if df.dim == 1 else f.DiffT(v, gradin * df))
            for f, df in zip(self.children, derivatives)
        )

    #def Derivative(self):
    #    pass

    @property
    def is_chunkable(self):
        return all(child.is_chunkable for child in self.children)

    def chunked_version(self, dimchk):
        return type(self)(
            *(
                (child if child.dim == 1 else child.chunked_version(dimchk))
                for child in self.children
            ),
            params=self.params
        )

    def chunked_vars(self, cat):
        res = set()
        for child in self.children:
            if child.dim != 1:
                res = set.union(res, set(child.chunked_vars(cat)))
        return list(res)

    def notchunked_vars(self, cat):
        res = set()
        for child in self.children:
            if child.dim == 1:
                res = set.union(res, set(child.Vars(cat)))
            else:
                res = set.union(res, set(child.notchunked_vars(cat)))
        return list(res)

    enable_test = True

class FusedVectorizedScalarOp(VectorizedScalarOp):

    string_id = "fused_vectorized"

    def recursive_str(self):
        return self.parent_op.recursive_str()

    def __init__(self, *args, params):
        parent_op, ind_child_op = params
        child_op = parent_op.children[ind_child_op]
        super().__init__(*args, params=params)
        self.parent_op, self.ind_child_op, self.child_op = parent_op, ind_child_op, child_op
    
    def ScalarOp(self, out, *args):
        i, m = self.ind_child_op, len(self.child_op.children)
        args_child = args[i:i+m]
        out_child = c_variable(out.dtype)
        str_child = self.child_op.ScalarOp(out_child, *args_child)
        args_parent = args[:i] + (out_child,) + args[i+m:]
        str_parent = self.parent_op.ScalarOp(out, *args_parent)
        return "{" + out_child.declare() + str_child + str_parent + "}"

    def DiffT(self, v, gradin):
        return self.parent_op.DiffT(v, gradin)