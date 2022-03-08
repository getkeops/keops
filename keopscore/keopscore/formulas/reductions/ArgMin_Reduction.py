from keopscore.utils.code_gen_utils import VectCopy
from keopscore.formulas.reductions.Min_ArgMin_Reduction_Base import (
    Min_ArgMin_Reduction_Base,
)
from keopscore.formulas.reductions.Zero_Reduction import Zero_Reduction


class ArgMin_Reduction(Min_ArgMin_Reduction_Base):
    """Implements the argmin reduction operation : for each i or each j, find the index of the
    minimal value of Fij operation is vectorized: if Fij is vector-valued, argmin is computed for each dimension."""

    string_id = "ArgMin_Reduction"

    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = formula.dim

    def FinalizeOutput(self, acc, out, i):
        acc_val, acc_ind = acc.split(self.dim, self.dim)
        return VectCopy(out, acc_ind)

    def DiffT(self, v, gradin):
        return Zero_Reduction(v.dim, v.cat % 2)
