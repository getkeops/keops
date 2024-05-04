from keopscore.formulas.Operation import Operation
from keopscore.formulas.maths.Extract import Extract
from keopscore.utils.meta_toolbox import c_empty_instruction


############################
######    Concat       #####
############################


class Concat(Operation):
    string_id = "Concat"
    print_fun = lambda *args: "[" + ",".join(str(arg) for arg in args) + "]"
    print_level = 9
    linearity_type = "all"

    def __init__(self, *args):
        super().__init__(*args)
        self.dim = sum(arg.dim for arg in args)

    def Op(self, out, table, *args):
        outs = out.split(*(arg.dim for arg in args))
        return sum((out.copy(arg) for out,arg in zip(outs,args)), start=c_empty_instruction)

    def DiffT_fun(self, v, gradin):
        from keopscore.formulas import Zero
        curr_dim = 0
        res = Zero(v.dim)
        for f in self.children:
            res += f.DiffT(v, Extract(gradin, curr_dim, f.dim))
            curr_dim += f.dim
        return res

    # parameters for testing the operation (optional)
    enable_test = True  # enable testing for this operation
    nargs = 2  # number of arguments
    test_argdims = [5, 3]  # dimensions of arguments for testing
    torch_op = None
