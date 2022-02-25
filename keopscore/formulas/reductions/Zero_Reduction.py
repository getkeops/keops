from keopscore.formulas.variables import Zero
from keopscore.formulas.reductions.Reduction import Reduction


class Zero_Reduction(Reduction):
    """Implements the zero reduction operation (fills output with zeros).
    N.B. The actual code for filling zeros is not here ; when a Zero_reduction is detected,
    the map_reduce scheme is redirected to CpuAssignZero or GpuAssignZero"""

    string_id = "Zero_Reduction"

    def __init__(self, dim, tagIJ):
        super().__init__(Zero(dim), tagIJ)
        self.dim = dim

    def DiffT(self, v, gradin, f0=None):
        return Zero_Reduction(v.dim, v.cat % 2)
