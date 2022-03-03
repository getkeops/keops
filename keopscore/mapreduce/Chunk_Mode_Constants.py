from keopscore.config.chunks import dimchunk
from keopscore.utils.code_gen_utils import GetDims, GetInds, Var_loader


class Chunk_Mode_Constants:
    def __init__(self, red_formula):

        varloader = Var_loader(red_formula)

        self.red_formula = red_formula
        self.dimred = red_formula.dimred  # dimension of reduction operation
        self.dimsp = varloader.dimsp  # dimensions of parameters variables
        self.indsp = varloader.indsp
        self.dimp = varloader.dimp
        self.dimout = (
            red_formula.dim
        )  # dimension of output variable of reduction operation
        formula = red_formula.formula
        self.dimfout = formula.dim  # dimension of output variable of inner function

        chunked_formula = formula.chunked_formulas(dimchunk)[0]
        self.dim_org = chunked_formula["dim_org"]
        self.nchunks = 1 + (self.dim_org - 1) // dimchunk
        self.dimlastchunk = self.dim_org - (self.nchunks - 1) * dimchunk
        self.nminargs = varloader.nminargs
        self.fun_chunked = chunked_formula["formula"]
        self.dimout_chunk = self.fun_chunked.dim

        self.varsi_chunked = self.fun_chunked.chunked_vars(red_formula.tagI)
        self.dimsx_chunked = GetDims(self.varsi_chunked)
        self.indsi_chunked = GetInds(self.varsi_chunked)

        self.varsj_chunked = self.fun_chunked.chunked_vars(red_formula.tagJ)
        self.dimsy_chunked = GetDims(self.varsj_chunked)
        self.indsj_chunked = GetInds(self.varsj_chunked)

        self.fun_postchunk = formula.post_chunk_formula(self.nminargs)

        self.varsi_postchunk = self.fun_postchunk.Vars(red_formula.tagI)

        self.dimsx_postchunk = GetDims(self.varsi_postchunk)
        self.indsi_postchunk = GetInds(self.varsi_postchunk)

        self.varsj_postchunk = self.fun_postchunk.Vars(red_formula.tagJ)
        self.dimsy_postchunk = GetDims(self.varsj_postchunk)
        self.indsj_postchunk = GetInds(self.varsj_postchunk)

        self.varsi_notchunked = list(
            set.union(
                set(self.varsi_postchunk),
                set(self.fun_chunked.notchunked_vars(red_formula.tagI)),
            )
        )

        # Here we detect if chunked variables are also used in the postchunk formula
        # Currently the code in GpuReduc1D_chunks.py does not handle this case, so
        # we will use the "chunk_postchunk_mix" tag defined below
        # in get_keops_dll to disable the chunked mode for the formula.
        self.chunk_postchunk_mix = (
            len(set.intersection(set(self.indsi_postchunk), set(self.indsi_chunked)))
            + len(set.intersection(set(self.indsj_postchunk), set(self.indsj_chunked)))
        ) > 0

        self.indsi_notchunked = GetInds(self.varsi_notchunked)
        self.dimsx_notchunked = GetDims(self.varsi_notchunked)
        self.dimx_notchunked = sum(self.dimsx_notchunked)

        self.varsj_notchunked = list(
            set.union(
                set(self.varsj_postchunk),
                set(self.fun_chunked.notchunked_vars(red_formula.tagJ)),
            )
        )
        self.indsj_notchunked = GetInds(self.varsj_notchunked)
        self.dimsy_notchunked = GetDims(self.varsj_notchunked)
        self.dimy_notchunked = sum(self.dimsy_notchunked)

        self.fun_lastchunked = formula.chunked_formulas(self.dimlastchunk)[0]["formula"]

        self.varsi_lastchunked = self.fun_lastchunked.chunked_vars(red_formula.tagI)
        self.indsi_lastchunked = GetInds(self.varsi_lastchunked)
        self.dimsx_lastchunked = GetDims(self.varsi_lastchunked)

        self.varsj_lastchunked = self.fun_lastchunked.chunked_vars(red_formula.tagJ)
        self.indsj_lastchunked = GetInds(self.varsj_lastchunked)
        self.dimsy_lastchunked = GetDims(self.varsj_lastchunked)

        self.varsi = [*self.varsi_notchunked, *self.varsi_chunked]
        self.dimsx = GetDims(self.varsi)
        self.indsi = GetInds(self.varsi)
        self.dimx = sum(self.dimsx)

        self.varsj = [*self.varsj_notchunked, *self.varsj_chunked]
        self.dimsy = GetDims(self.varsj)
        self.indsj = GetInds(self.varsj)
        self.dimy = sum(self.dimsy)

        self.inds = [*self.indsi, *self.indsj, *self.indsp]

        self.varsi_last = [*self.varsi_notchunked, *self.varsi_lastchunked]
        self.indsi_last = GetInds(self.varsi_last)
        self.dimsx_last = GetDims(self.varsi_last)

        self.varsj_last = [*self.varsj_notchunked, *self.varsj_lastchunked]
        self.indsj_last = GetInds(self.varsj_last)
        self.dimsy_last = GetDims(self.varsj_last)
