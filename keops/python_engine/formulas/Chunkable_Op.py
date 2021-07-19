from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.formulas.variables.Var import Var
from keops.python_engine import enable_chunk, dim_treshold_chunk, specdim_use_chunk1, specdim_use_chunk2, specdim_use_chunk3, specdim_use_chunk4

class Chunkable_Op(Operation):
    
    def chunked_version(self, dimchk):
        chunked_args = [child.chunked_version(dimchk) for child in self.children]
        return type(self)(*chunked_args, params=self.params)
    
    def chunked_vars(self, cat):
        res = set()
        for child in self.children:
            res = set.union(res, set(child.chunked_vars(cat)))
        return list(res)

    def notchunked_vars(self, cat):
        return []

    @property
    def use_chunk(self):
        test = enable_chunk & all(child.is_chunkable for child in self.children)
        child = self.children[0]
        subtest = (child.dim > dim_treshold_chunk)
        subtest |= (child.dim == specdim_use_chunk1) or (child.dim == specdim_use_chunk2)
        subtest |= (child.dim == specdim_use_chunk3) or (child.dim == specdim_use_chunk4)
        test &= subtest
        return test
    
    def chunked_formula(self, dimchk):
        if self.use_chunk:
            return dict(formula=self.chunked_version(dimchk), dim_org=self.children[0].dim)
        else:
            return None

    def chunked_formulas(self, dimchk):
        chk_f = self.chunked_formula(dimchk)
        res = [chk_f] if chk_f is not None else []
        for child in self.children:
            res += child.chunked_formulas(dimchk)
        return res

    @property
    def num_chunked_formulas(self):
        return sum(child.num_chunked_formulas for child in self.children) + self.use_chunk

    def post_chunk_formula(self, ind):
        if self.use_chunk:
            return Var(ind, 1, 3)
        else:
            return type(self)(*(child.post_chunk_formula(ind) for child in self.children), params=self.params)
