from keopscore.formulas.Operation import Operation
from keopscore.utils.meta_toolbox import c_zero_float
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.unique_object import unique_object

# //////////////////////////////////////////////////////////////
# ////     VECTOR "INJECTION" : ExtractT<F,START,DIM>       ////
# //////////////////////////////////////////////////////////////


def ExtractT(f, start, dim):
    return ExtractT_Impl_Factory(start, dim)(f)


class ExtracT_Impl(Operation):

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


class ExtractT_Impl_Factory(metaclass=unique_object):

    def __init__(self, start, dim):

        class Class(ExtracT_Impl):

            string_id = "ExtractT"
            linearity_type = "all"

            def __init__(self, f):
                if start + f.dim > dim or start < 0:
                    KeOps_Error("Index out of bound in ExtractT")
                super().__init__(f)
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
