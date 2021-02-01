import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'keops', 'python_engine'))
from operations import *
from reductions import *  
from link_compile import *     
import time 
        
class LoadKeOps:
    def __init__(
        self, formula, aliases, dtype, lang, optional_flags=[], include_dirs=[]
    ):
        self.aliases = aliases
        self.dtype = dtype
        self.lang = lang
        self.optional_flags = optional_flags
        self.include_dirs = include_dirs
        self.red_formula = eval(formula)
        self.dimout = self.red_formula.dim
        self.tagIJ = self.red_formula.tagI
    
    def genred_pytorch(self, tagCPUGPU, tag1D2D, tagHostDevice, device_id, ranges, nx, ny, *args):
        
        import torch
        nargs = len(args)
        dtype = args[0].dtype
        device = args[0].device
        if dtype == torch.float32:
            c_dtype = "float"
        elif dtype == torch.float64:
            c_dtype = "double"
        else:
            raise ValueError("not implemented")
        
        if '-D__TYPEACC__=double' in self.optional_flags:
            c_dtype_acc = 'double'
        elif '-D__TYPEACC__=float' in self.optional_flags:
            c_dtype_acc = 'float'
        else:
            raise ValueError('not implemented')
            
        if '-DSUM_SCHEME=0':
            sum_scheme = 'direct_sum'
        elif '-DSUM_SCHEME=1':
            sum_scheme = 'block_sum'
        elif '-DSUM_SCHEME=2':
            sum_scheme = 'kahan_scheme'
        else:
            raise ValueError('not implemented')
            
        if device.type == "cpu":
            myred = CpuReduc(self.red_formula, c_dtype, c_dtype, nargs, sum_scheme=sum_scheme)
        else:
            myred = GpuReduc1D(self.red_formula, c_dtype, c_dtype, nargs, sum_scheme=sum_scheme)
        if not os.path.exists(myred.dllname):
            print("compiling dll...", end="", flush=True)
            start = time.time()
            myred.compile_code()
            elapsed = time.time()-start
            print("done ({:.2f} s)".format(elapsed))
        M, N = (nx, ny) if self.red_formula.tagI==0 else (ny, nx)
        out = torch.zeros(M, self.red_formula.dim, dtype=dtype, device=device)
        if self.red_formula.formula != Zero(self.red_formula.dim):
            myred(M, N, out, *args)
        return out
        
    def import_module(self):
        return self
        