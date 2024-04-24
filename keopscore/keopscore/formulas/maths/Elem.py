from keopscore.formulas.Operation import Operation
from keopscore.utils.meta_toolbox.c_instruction import c_empty_instruction
from keopscore.utils.meta_toolbox.c_lvalue import c_value
from keopscore.utils.misc_utils import KeOps_Error

###################################################################
######    ELEMENT EXTRACTION : Elem(f,m) (aka get_item)       #####
###################################################################


class Elem(Operation):
    string_id = "Elem"
    print_spec = ("[", "]"), "item", 1
    linearity_type = "all"

    def __init__(self, f, m=None, params=None):
        # N.B. init via params keyword is used for compatibility with base class.
        if m is None:
            # here params should be a tuple containing one single integer
            (m,) = params
        super().__init__(f, params=(m,))
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

    # parameters for testing the operation (optional)
    enable_test = True  # enable testing for this operation
    nargs = 1  # number of arguments
    test_argdims = [5]  # dimensions of arguments for testing
    test_params = [3]  # dimensions of parameters for testing
    torch_op = "lambda x, m : x[..., m][..., None]"
