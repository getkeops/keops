from keops.python_engine import dimchunk
from keops.python_engine.utils.code_gen_utils import GetDims, GetInds

class Chunk_Mode_Constants:
    
    def __init__(self, red_formula):
        self.red_formula = red_formula
        self.dimred = red_formula.dimred        # dimension of reduction operation
        self.dimsp = red_formula.dimsp         # dimensions of parameters variables         
        self.indsp = red_formula.indsp 
        self.dimp = red_formula.dimp
        self.dimout = red_formula.dim           # dimension of output variable of reduction operation
        formula = red_formula.formula
        self.dimfout = formula.dim  # dimension of output variable of inner function
        chunked_formula = formula.chunked_formulas(dimchunk)[0]
        self.dim_org = chunked_formulas["dim_org"]
        self.nchunks = 1 + (self.dim_org-1) / dimchunk
        self.dimlastchunk = self.dim_org - (self.nchunks-1)*dimchunk
        self.nminargs = formula.nminargs
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
        
        self.varsj_postchunk = self.fun_postchunk.Vars(red_formula.tagJ)
        self.dimsy_postchunk = GetDims(self.varsj_postchunk)
        
        self.varsi_notchunked = set.union(self.varsi_postchunk, self.fun_chunked.notchunked_vars(red_formula.tagI))
        self.indsi_notchunked = GetInds(self.varsi_notchunked)
        self.dimsx_notchunked = GetDims(self.varsi_notchunked)
        self.dimx_notchunked = sum(self.dimsx_notchunked)
        
        self.varsj_notchunked = set.union(self.varsj_postchunk, self.fun_chunked.notchunked_vars(red_formula.tagJ))
        self.indsj_notchunked = GetInds(self.varsj_notchunked)
        self.dimsy_notchunked = GetDims(self.varsj_notchunked)
        self.dimy_notchunked = sum(self.dimsy_notchunked)

        self.fun_lastchunked = formula.chunked_formulas(self.dimalastchunk)[0]["formula"]
        
        self.varsi_lastchunked = self.fun_lastchunked.chunked_vars(red_formula.tagI)
        self.dimsx_lastchunked = GetDims(self.varsi_lastchunked)
        
        self.varsj_lastchunked = self.fun_lastchunked.chunked_vars(red_formula.tagJ)
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
        self.dimsx_last = GetDims(self.varsi_last)
        
        self.varsj_last = [*self.varsj_notchunked, *self.varsj_lastchunked]
        self.dimsy_last = GetDims(self.varsj_last)


def Get_DIMY_SHARED(red_formula, use_chunk_mode):
    if use_chunk_mode==0:
        return sum(red_formula.dimsy)
    elif use_chunk_mode==1:
        return Chunk_Mode_Constants(red_formula).dimy
template < class FUN, int USE_CHUNK_MODE > struct Get_DIMY_SHARED;

template < class FUN >
struct Get_DIMY_SHARED<FUN,0> {
    static const int Value = FUN::DIMSY::SUM;
};

template < class FUN >
struct Get_DIMY_SHARED<FUN,1> {
    static const int Value = Chunk_Mode_Constants<FUN>::DIMY;
};

template < class FUN >
struct Get_DIMY_SHARED<FUN,2> {
    static const int Value = DIMFINALCHUNK;
};

