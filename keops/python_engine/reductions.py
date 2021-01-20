from tree import tree
from utils import GetDims, GetInds, c_variable, VectAssign, VectApply, VectCopy, cast_to

class Reduction(tree):
    def __init__(self, formula, tagI):
        self.formula = formula
        self.children = [formula]
        self.tagI = tagI
        self.tagJ = 1-tagI
        self.cat = tagI
        self.Varsi = formula.Vars(cat=tagI)
        self.indsi = GetInds(self.Varsi)
        self.dimsx = GetDims(self.Varsi)
        self.dimx = sum(self.dimsx)
        self.Varsj = formula.Vars(cat=self.tagJ)
        self.indsj = GetInds(self.Varsj)
        self.dimsy = GetDims(self.Varsj)
        self.dimy = sum(self.dimsy)
        self.Varsp = formula.Vars(cat=2)
        self.indsp = GetInds(self.Varsp)
        self.dimsp = GetDims(self.Varsp)
        self.dimp = sum(self.dimsp)

class Sum_Reduction(Reduction):
    string_id = "Sum_Reduction"
    def __init__(self, formula, tagIJ):
        super().__init__(formula, tagIJ)
        self.dim = formula.dim
        self.dimred = self.dim
    def InitializeReduction(self, tmp):
        zero = c_variable("0.0f","float")
        return VectAssign(tmp, self.dim, zero)
    def ReducePairScalar(self, tmp, xi):
        return f"{tmp()} += {cast_to(tmp.dtype)}({xi()});"
    def ReducePairShort(self, tmp, xi, val):
        return VectApply(self.ReducePairScalar,self.dim, self.dim, tmp, xi)
    def FinalizeOutput(self, acc, out, i):
        return VectCopy(self.dim, out, acc)
        

