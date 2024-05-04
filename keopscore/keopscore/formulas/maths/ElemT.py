from keopscore.formulas.Operation import Operation
from keopscore.utils.meta_toolbox import c_zero_float, c_for_loop
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.unique_object import unique_object

#####################################################
######    ELEMENT "INJECTION" : ElemT(f,n,m)   ######
#####################################################


def ElemT(f, n, m):
    return ElemT_Impl_Factory(n, m)(f)


class ElemT_Impl(Operation):

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


class ElemT_Impl_Factory(metaclass=unique_object):

    def __init__(self, n, m):

        class Class(ElemT_Impl):

            string_id = "ElemT"
            linearity_type = "all"

            def __init__(self, f):
                super().__init__(f)
                if f.dim != 1:
                    KeOps_Error("Input of ElemT should be a scalar")
                self.dim = n
                self.n = n
                self.m = m

            def Op(self, out, table, arg):
                n, m = self.n, self.m
                loop1, k = c_for_loop(0, m, 1, pragma_unroll=True)
                res = loop1(out[k].assign(c_zero_float))
                res += out[m].assign(arg.value)
                loop2, k = c_for_loop(m + 1, n, 1, pragma_unroll=True)
                res += loop2(out[k].assign(c_zero_float))
                return res

            def GradFun(self, v, gradin):
                from keopscore.formulas.maths.Elem import Elem

                f = self.children[0]
                return f.DiffT(v, Elem(gradin, self.m))

        self.Class = Class

    def __call__(self, f):
        return self.Class(f)
