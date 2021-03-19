from pykeops.common.get_keops_routine import get_keops_routine    
import time 
        
class LoadKeOps_new:
    
    def __init__(
        self, formula, aliases, tagCPUGPU, dtype, lang, args, optional_flags=[], include_dirs=[]
    ):
        
        if lang == "torch":
            from pykeops.torch.utils import torchtools
            tools = torchtools
        elif lang == "numpy":
            from pykeops.numpy.utils import numpytools
            tools = numpytools
            
        nargs = len(args)
        
        self.dtype = tools.dtype(args[0])
        dtypename = tools.dtypename(self.dtype)
        
        if dtypename == "float32":
            c_dtype = "float"
        elif dtypename == "float64":
            c_dtype = "double"
        else:
            raise ValueError("not implemented")
            
        if '-D__TYPEACC__=double' in optional_flags:
            c_dtype_acc = 'double'
        elif '-D__TYPEACC__=float' in optional_flags:
            c_dtype_acc = 'float'
        else:
            raise ValueError('not implemented')
        
        if '-DSUM_SCHEME=0' in optional_flags:
            sum_scheme = 'direct_sum'
        elif '-DSUM_SCHEME=1' in optional_flags:
            sum_scheme = 'block_sum'
        elif '-DSUM_SCHEME=2' in optional_flags:
            sum_scheme = 'kahan_scheme'
        else:
            print(optional_flags)
            raise ValueError('not implemented')
        
        aliases_new = []
        for k, alias in enumerate(aliases):
            alias = alias.replace(" ","")
            if "=" in alias:
                varname, var = alias.split("=")
                if "Vi" in var:
                    cat = 0
                elif "Vj" in var:
                    cat = 1
                elif "Pm" in var:
                    cat = 2
                dim = eval(var[3:-1])
                alias = f"{varname}=Var({k},{dim},{cat})"
                aliases_new.append(alias)
        aliases = aliases_new
        
        map_reduce_id = "CpuReduc" if tagCPUGPU==0 else "GpuReduc1D" 

        self.myfun = get_keops_routine(map_reduce_id, formula, aliases, nargs, c_dtype, c_dtype_acc, sum_scheme)

        self.dimout = self.myfun.dim
        self.tagIJ = self.myfun.tagI
        
        self.tools = tools
    
    def __call__(self, tag1D2D, tagHostDevice, device_id, ranges, nx, ny, *args):
        
        device = self.tools.device_dict(args[0])
        if device_id == -1:
            device_id = device["index"]
            
        M, N = (nx, ny) if self.myfun.tagI==0 else (ny, nx)
        out = self.tools.zeros((M, self.myfun.dim), dtype=self.dtype, device=device)
        out_ctype = self.tools.ctypes(out)
        args_ctype = (self.tools.ctypes(arg) for arg in args)
        self.myfun(M, N, device_id, out_ctype, *args_ctype)
        return out  

        
