from keopscore.formulas.Operation import Operation
from keopscore.utils.meta_toolbox import c_zero_float
from keopscore.utils.misc_utils import KeOps_Error

# //////////////////////////////////////////////////////////////
# ////     VECTOR "INJECTION" : ExtractT<F,START,DIM>       ////
# //////////////////////////////////////////////////////////////


class ExtractT(Operation):
    string_id = "ExtractT"
    linearity_type = "all"

    def __init__(self, f, start=None, dim=None, params=None):
        # N.B. init via params keyword is used for compatibility with base class.
        if start is None:
            # here we assume dim is also None and params is be a tuple containing start and dim
            start, dim = params
        if start + f.dim > dim or start < 0:
            KeOps_Error("Index out of bound in ExtractT")
        super().__init__(f, params=(start, dim))
        self.start = start
        self.dim = dim
        self.dimarg = f.dim

    def Op(self, out, table, arg):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        out_prev, out_mid, out_end = out.split(
            self.start, self.dimarg, self.dim - self.start - self.dimarg
        )
        return (
            out_prev.assign(c_zero_float)
            + out_mid.copy(arg)
            + out_end.assign(c_zero_float)
        )

    def DiffT(self, v, gradin):
        from keopscore.formulas.maths.Extract import Extract

        f = self.children[0]
        return f.DiffT(v, Extract(gradin, self.start, f.dim))

    # parameters for testing the operation (optional)
    enable_test = True  # enable testing for this operation
    nargs = 1  # number of arguments
    test_argdims = [5]  # dimensions of arguments for testing
    test_params = [3, 10]  # dimensions of parameters for testing

    def torch_op(x, s, d):  # equivalent PyTorch operation
        import torch

        out = torch.zeros((*x.shape[:-1], d), device=x.device, dtype=x.dtype)
        out[..., s : (s + x.shape[-1])] = x
        return out
