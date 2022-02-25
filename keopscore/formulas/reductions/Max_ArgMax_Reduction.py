from keopscore.utils.code_gen_utils import VectCopy
from keopscore.formulas.reductions.Max_ArgMax_Reduction_Base import (
    Max_ArgMax_Reduction_Base,
)


class Max_ArgMax_Reduction(Max_ArgMax_Reduction_Base):
    """Implements the max+argmax reduction operation : for each i or each j, find the maximal value of Fij and its index
    operation is vectorized: if Fij is vector-valued, max+argmax is computed for each dimension."""

    string_id = "Max_ArgMax_Reduction"

    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = 2 * formula.dim

    def FinalizeOutput(self, acc, out, i):
        return VectCopy(out, acc)
