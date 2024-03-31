from keopscore.formulas.Operation import Operation
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.misc_utils import KeOps_Error

#########################
## I and J placeholders
#########################


class I(Operation):
    """I operation class. I() is a symbolic
    object that encodes the "i" index"""

    string_id = "I"

    def __init__(self, params=()):
        # N.B. init via params keyword is used for compatibility with base class.
        if params is not ():
            KeOps_Error("There should be no parameter for I operation.")
        super().__init__(params=())
        self.dim = 1

    def DiffT(self, v, gradin):
        return Zero(v.dim)


class J(Operation):
    """J operation class. J() is a symbolic
    object that encodes the "j" index"""

    string_id = "J"

    def __init__(self, params=()):
        # N.B. init via params keyword is used for compatibility with base class.
        if params is not ():
            KeOps_Error("There should be no parameter for J operation.")
        super().__init__(params=())
        self.dim = 1

    def DiffT(self, v, gradin):
        return Zero(v.dim)
