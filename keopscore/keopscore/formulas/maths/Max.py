from keopscore.formulas.Operation import Operation
from keopscore.formulas.maths.ArgMax import ArgMax
from keopscore.formulas.maths.OneHot import OneHot
from keopscore.utils.code_gen_utils import c_for_loop, c_if, value
from keopscore.utils.misc_utils import KeOps_Error

############################
######    Max       #####
############################


class Max(Operation):
    string_id = "Max"

    def __init__(self, f):
        super().__init__(f)
        if f.dim < 1:
            KeOps_Error("Max operation is only possible when dimension is non zero.")
        self.dim = 1
        self.argdim = f.dim

    def Op(self, out, table, arg):
        loop, k = c_for_loop(1, arg.dim, 1, pragma_unroll=True)
        string = value(out).assign(arg[0])
        if out.dtype == "half2":
            loop_string = f"""
                // we have to work element-wise...
                __half2 cond = __hlt2(*{out.id},{arg[k].id});                       // cond = (out > outF[k]) (element-wise)
                __half2 negcond = __float2half2_rn(1.0f)-cond;                      // negcond = 1-cond
                *{out.id} = cond * {arg[k].id} + negcond * *{out.id};               // out  = cond * outF[k] + (1-cond) * out
                            """
            string += loop(loop_string)
        else:
            string += loop(c_if(arg[k] > value(out), value(out).assign(arg[k])))
        return string

    def DiffT(self, v, gradin):
        f = self.children[0]
        return f.DiffT(v, OneHot(ArgMax(f), self.argdim) * gradin)

    # parameters for testing the operation (optional)
    enable_test = True  # enable testing for this operation
    nargs = 1  # number of arguments
    test_argdims = [5]  # dimensions of arguments for testing
    torch_op = "lambda x : torch.max(x, dim=-1, keepdim=True)[0].type(x.dtype)"
