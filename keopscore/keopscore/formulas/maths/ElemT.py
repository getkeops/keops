from keopscore.formulas.Operation import Operation
from keopscore.utils.meta_toolbox import c_zero_float, c_for_loop
from keopscore.utils.misc_utils import KeOps_Error

#####################################################
######    ELEMENT "INJECTION" : ElemT(f,n,m)   ######
#####################################################


class ElemT(Operation):
    string_id = "ElemT"
    linearity_type = "all"

    def __init__(self, f, n=None, m=None, params=None):
        # N.B. init via params keyword is used for compatibility with base class.
        if n is None:
            # here we assume m is also None and params is be a tuple containing n and m
            n, m = params
        super().__init__(f, params=(n, m))
        if f.dim != 1:
            KeOps_Error("Input of ElemT should be a scalar")
        self.dim = n
        self.n = n
        self.m = m

    def Op(self, out, table, arg):
        n, m = self.n, self.m
        loop1, k = c_for_loop(0, m, 1, pragma_unroll=True)
        string = loop1(out[k].assign(c_zero_float))
        string += out[m].assign(arg.value)
        loop2, k = c_for_loop(m + 1, n, 1, pragma_unroll=True)
        string += loop2(out[k].assign(c_zero_float))
        return string

    def DiffT(self, v, gradin):
        from keopscore.formulas.maths.Elem import Elem

        f = self.children[0]
        return f.DiffT(v, Elem(gradin, self.m))

    # parameters for testing the operation (optional)
    enable_test = True  # enable testing for this operation
    nargs = 1  # number of arguments
    test_argdims = [1]  # dimensions of arguments for testing
    test_params = [5, 3]  # dimensions of parameters for testing

    def torch_op(x, n, m):  # equivalent PyTorch operation
        import torch

        out = torch.zeros((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
        out[..., m][..., None] = x
        return out
