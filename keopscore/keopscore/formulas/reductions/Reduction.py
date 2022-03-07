from keopscore.utils.code_gen_utils import VectApply, VectCopy
from keopscore.utils.Tree import Tree


class Reduction(Tree):
    """Base class for all KeOps final reductions over a formula"""

    def __init__(self, formula, tagI):
        """- formula is an object of type Operation, it is the formula on which we apply a reduction
        - tagI : 0 or 1, specifies wether we do the reduction over "i"-indexed or "j"-indexed variables."""

        # We initialize several constants, most of them infered from the formula
        self.formula = formula
        self.children = [formula]
        self.params = (tagI,)
        self.tagI = tagI
        self.tagJ = 1 - tagI
        self.cat = tagI
        self.Vars_ = formula.Vars_

    def ReducePair(self, acc, xi):
        """Returns C++ code that implements the update phase of the reduction.
        by default it consists in a vectorized version of the ReducePairScalar operation."""
        return VectApply(self.ReducePairScalar, acc, xi)

    def ReducePairShort(self, acc, xi, ind):
        # N.B next lines are useless here, but to be used in other reductions :
        # if xi.dtype == "half2":
        #     half2_val = c_variable("half2_ind")
        #     string = half2_val.declare_assign(f"__floats2half2_rn(2*{ind()},2*{ind()}+1)")
        return self.ReducePair(acc, xi)

    def FinalizeOutput(self, acc, out, i):
        """Returns C++ code that implements the final output of the reduction.
        For most reducitons it is a simple copy of the temporary variable
        updated during the reduction, with possibly a cast if the accumulator was of
        different data type."""
        return VectCopy(out, acc)
