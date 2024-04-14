from keopscore.formulas.Chunkable_Op import Chunkable_Op
from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.utils.code_gen_utils import VectApply, c_variable


class InnerReductionOp(Chunkable_Op):

    def __new__(cls, *args, allow_fuse=True, **kwargs):
        obj = super(InnerReductionOp, cls).__new__(cls)
        obj.__init__(*args, **kwargs)
        args = obj.children
        if allow_fuse:
            for ind, arg in enumerate(args):
                if isinstance(arg, VectorizedScalarOp):
                    fused_args = args[:ind] + arg.children + args[ind + 1 :]
                    return FusedInnerReductionOp.__new__(
                        FusedInnerReductionOp,
                        *fused_args,
                        params=(obj, ind),
                        allow_fuse=False
                    )
        return obj

    def Op(self, out, table, *args):
        return out.assign(self.init_red) + VectApply(self.ScalarOp, out, *args)


class FusedInnerReductionOp(InnerReductionOp):

    string_id = "fused_reduction"

    dim = 1

    def recursive_str(self):
        return self.parent_op.recursive_str()

    def __init__(self, *args, params=()):
        parent_op, ind_child_op = params
        child_op = parent_op.children[ind_child_op]
        super().__init__(*args, params=params)
        self.parent_op, self.ind_child_op, self.child_op = (
            parent_op,
            ind_child_op,
            child_op,
        )
        self.init_red = parent_op.init_red

    def ScalarOp(self, out, *args):
        i, m = self.ind_child_op, len(self.child_op.children)
        args_child = args[i : i + m]
        out_child = c_variable(out.dtype)
        str_child = self.child_op.ScalarOp(out_child, *args_child)
        args_parent = args[:i] + (out_child,) + args[i + m :]
        str_parent = self.parent_op.ScalarOp(out, *args_parent)
        return "{" + out_child.declare() + str_child + str_parent + "}"

    def DiffT(self, v, gradin):
        return self.parent_op.DiffT(v, gradin)

    def initacc_chunk(self, acc):
        return self.parent_op.initacc_chunk(acc)

    def acc_chunk(self, acc, out):
        return self.parent_op.acc_chunk(acc, out)
