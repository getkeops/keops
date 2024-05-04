from keopscore.formulas.Operation import Operation
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.utils.meta_toolbox import c_empty_instruction, c_array_scalar

#########################
## I and J placeholders
#########################


class I(Operation):
    """I operation class. I() is a symbolic
    object that encodes the "i" index"""

    string_id = "I"

    def __init__(self):
        super().__init__()
        self.dim = 1

    def get_code_and_expr(self, dtype, table, i, j, tagI):
        out = i if tagI == 0 else j
        return c_empty_instruction, c_array_scalar(out)

    def GradFun(self, v, gradin):
        return Zero(v.dim)


class J(Operation):
    """J operation class. J() is a symbolic
    object that encodes the "j" index"""

    string_id = "J"

    def __init__(self):
        super().__init__()
        self.dim = 1

    def get_code_and_expr(self, dtype, table, i, j, tagI):
        out = j if tagI == 0 else i
        return c_empty_instruction, c_array_scalar(out)

    def GradFun(self, v, gradin):
        return Zero(v.dim)
