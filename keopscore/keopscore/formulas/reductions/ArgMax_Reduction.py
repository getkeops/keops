from keopscore.formulas.reductions.Max_ArgMax_Reduction_Base import (
    Max_ArgMax_Reduction_Base,
)
from keopscore.formulas.reductions.Zero_Reduction import Zero_Reduction


class ArgMax_Reduction(Max_ArgMax_Reduction_Base):
    """Implements the argmax reduction operation : for each i or each j, find the index of the
    maximal value of Fij operation is vectorized: if Fij is vector-valued, argmax is computed for each dimension.
    """

    string_id = "ArgMax_Reduction"

    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = formula.dim

    def FinalizeOutput(self, acc, out, i):
        acc_val, acc_ind = acc.split(self.dim, self.dim)
        return out.copy(acc_ind)

    def GradFun(self, v, gradin, f0=None):
        return Zero_Reduction(v.dim, v.cat % 2)

    def Diff(self, v, diffin, f0=None):
        return Zero_Reduction(self.dim, self.tagI)
