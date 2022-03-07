from keopscore.utils.code_gen_utils import VectCopy
from keopscore.formulas.reductions.Min_ArgMin_Reduction_Base import (
    Min_ArgMin_Reduction_Base,
)


class Min_ArgMin_Reduction(Min_ArgMin_Reduction_Base):
    """Implements the min+argmin reduction operation : for each i or each j, find the minimal value of Fij
    and its index operation is vectorized: if Fij is vector-valued, min+argmain is computed for each dimension."""

    string_id = "Min_ArgMin_Reduction"

    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = 2 * formula.dim

    def FinalizeOutput(self, acc, out, i):
        return VectCopy(out, acc)
