from keops.python_engine.formulas.Operation import Operation
from keops.python_engine.formulas.variables.Var import Var
from keops.python_engine import enable_chunk, dim_treshold_chunk, specdim_use_chunk1, specdim_use_chunk2, specdim_use_chunk3, specdim_use_chunk4

class Chunkable_Op(Operation):
    
    def chunked_version(self, dimchk):
        chunked_args = [arg.chunked_version(dimchk) for arg in args]
        return type(self)(*chunked_args, params=self.params)
    
    def chunked_vars(self, cat):
        return set.union(*(child.chunked_vars(cat) for child in self.children)) if len(self.children) > 0 else set()

    def not_chunked_vars(self, cat):
        return set()

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
            return [[self.chunked_version(dimchk), [self.children[0].dim]]]
        else:
            return []

    def chunked_formulas(self, dimchk):
        return [*self.chunked_formula(dimchk), *(child.chunked_formulas(dimchk) for child in self.children)]

    @property
    def num_chunked_formulas(self):
        return sum(child.num_chunked_formulas for child in self.children) + self.use_chunk

    def post_chunk_formula(self, ind):
        if self.use_chunk:
            return Var(ind, 1, 3)
        else:
            return type(self)(*(child.post_chunk_formula(ind) for child in self.children), params=self.params)