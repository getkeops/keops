from keopscore.formulas.Operation import Operation
from keopscore.utils.meta_toolbox.c_instruction import c_empty_instruction
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.unique_object import unique_object

###################################################################
######    ELEMENT EXTRACTION : Elem(f,m) (aka get_item)       #####
###################################################################


def Elem(f, m):
    return Elem_Factory(m)(f)


class Elem_Impl(Operation):

    # parameters for testing the operation (optional)
    enable_test = True  # enable testing for this operation
    nargs = 1  # number of arguments
    test_argdims = [5]  # dimensions of arguments for testing
    test_params = [3]  # dimensions of parameters for testing
    torch_op = "lambda x, m : x[..., m][..., None]"


class Elem_Factory(metaclass=unique_object):

    def __init__(self, m):

        class Class(Elem_Impl):

            string_id = "Elem"
            print_fun = lambda x: f"{x}[{m}]"
            print_level = 1

            linearity_type = "all"

            def __init__(self, f):
                super().__init__(f)
                if f.dim <= m:
                    KeOps_Error("Index out of bound in Elem")
                self.dim = 1
                self.m = m

            def get_code_and_expr(self, dtype, table, i, j, tagI):
                (child,) = self.children
                code, code_elem, expr = child.get_code_and_expr_elem(
                    dtype, table, i, j, tagI, self.m
                )
                return code + code_elem, expr

            def get_code_and_expr_elem(self, dtype, table, i, j, tagI, elem):
                (child,) = self.children
                code, code_elem, expr = child.get_code_and_expr_elem(
                    dtype, table, i, j, tagI, self.m
                )
                return code, code_elem, expr

            def __call__(self, out, table, i, j, tagI):
                (child,) = self.children
                code, code_elem, expr = child.get_code_and_expr_elem(
                    out.dtype, table, i, j, tagI, self.m
                )
                return code + code_elem + out.assign(expr)

            def DiffT(self, v, gradin):
                from keopscore.formulas.maths.ElemT import ElemT

                f = self.children[0]
                return f.DiffT(v, ElemT(gradin, f.dim, self.m))

        self.Class = Class

    def __call__(self, f):
        return self.Class(f)
