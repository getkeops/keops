from keopscore.formulas.Operation import Operation
from keopscore.formulas.variables.Var import Var
from keopscore.config.chunks import enable_chunk, dim_treshold_chunk, specdims_use_chunk


class Chunkable_Op(Operation):
    def chunked_version(self, dimchk):
        chunked_args = [child.chunked_version(dimchk) for child in self.children]
        if self.params == ():
            return type(self)(*chunked_args)
        else:
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
        subtest = (child.dim >= dim_treshold_chunk) | (child.dim in specdims_use_chunk)
        test &= subtest
        return test

    def chunked_formula(self, dimchk):
        if self.use_chunk:
            return dict(
                formula=self.chunked_version(dimchk), dim_org=self.children[0].dim
            )
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
        return (
            sum(child.num_chunked_formulas for child in self.children) + self.use_chunk
        )

    def post_chunk_formula(self, ind):
        if self.use_chunk:
            return Var(ind, 1, 3)
        else:
            return type(self)(
                *(child.post_chunk_formula(ind) for child in self.children),
                params=self.params
            )
