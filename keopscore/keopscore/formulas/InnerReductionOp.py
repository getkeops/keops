from keopscore.formulas.Chunkable_Op import Chunkable_Op
from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.Operation import FusedOp
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


class FusedInnerReductionOp(FusedOp, InnerReductionOp):

    string_id = "fused_reduction"

    dim = 1

    def __init__(self, *args, params=()):
        super().__init__(*args, params=params)
        self.init_red = self.parent_op.init_red    

    def initacc_chunk(self, acc):
        return self.parent_op.initacc_chunk(acc)

    def acc_chunk(self, acc, out):
        return self.parent_op.acc_chunk(acc, out)
