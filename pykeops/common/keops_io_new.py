import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'keops', 'python_engine'))
from operations import *
from reductions import *  
from get_keops_routine import *     
import time 
import torch
        
class LoadKeOps:
    
    def __init__(
        self, formula, aliases, dtype, lang, optional_flags=[], include_dirs=[]
    ):
        for k, alias in enumerate(aliases):
            alias = alias.replace(" ","")
            if "=" in alias:
                varname, var = alias.split("=")
                if "Vi" in var:
                    dim = eval(var[3:-1])
                    var = f"Var({k},{dim},0)"
                elif "Vj" in var:
                    dim = eval(var[3:-1])
                    var = f"Var({k},{dim},1)"
                elif "Pm" in var:
                    dim = eval(var[3:-1])
                    var = f"Var({k},{dim},2)"
                formula = formula.replace(varname, var)

        #self.aliases = aliases
        #self.dtype = dtype
        #self.lang = lang
        self.optional_flags = optional_flags
        #self.include_dirs = include_dirs
        self.red_formula = eval(formula)
        self.dimout = self.red_formula.dim
        self.tagIJ = self.red_formula.tagI
    
    def genred_pytorch(self, tagCPUGPU, tag1D2D, tagHostDevice, device_id, ranges, nx, ny, *args):

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
            map_reduce_id = "CpuReduc"
        else:
            map_reduce_id = "GpuReduc1D"   
        myfun = get_keops_routine(map_reduce_id, self.red_formula, nargs, c_dtype, c_dtype_acc, sum_scheme)
        M, N = (nx, ny) if myfun.tagI==0 else (ny, nx)
        out = torch.zeros(M, myfun.dim, dtype=dtype, device=device)
        out_ptr = out.data_ptr()
        args_ptr = (arg.data_ptr() for arg in args)
        myfun(M, N, out_ptr, *args_ptr)
        return out
        
    def import_module(self):
        return self
        
