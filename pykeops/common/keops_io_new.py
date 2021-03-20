from pykeops.common.get_keops_routine import get_keops_routine    
import time 
        
class LoadKeOps_new:
    
    def __init__(
        self, formula, aliases, dtype, lang, optional_flags=[], include_dirs=[]
    ):
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
        self.aliases = aliases
        self.lang = lang
        self.optional_flags = optional_flags
        self.red_formula_string = formula
        self.dtype = dtype
    
    def genred(self, tagCPUGPU, tag1D2D, tagHostDevice, device_id_, ranges, nx, ny, *args):
        
        if self.lang == "torch":
            from pykeops.torch.utils import torchtools
            tools = torchtools
        elif self.lang == "numpy":
            from pykeops.numpy.utils import numpytools
            tools = numpytools
        
        nargs = len(args)
        device = tools.device_dict(args[0])
        dtype = tools.dtype(args[0])
        dtypename = tools.dtypename(dtype)
                
        if self.dtype not in ['auto', dtypename]:
            print("[KeOps] warning : setting a dtype argument in Genred different from the input dtype of tensors is not permitted anymore, argument is ignored.")
        
        if dtypename == "float32":
            c_dtype = "float"
        elif dtypename == "float64":
            c_dtype = "double"
        else:
            raise ValueError("not implemented")
        
        if '-D__TYPEACC__=double' in self.optional_flags:
            c_dtype_acc = 'double'
            self.optional_flags.remove('-D__TYPEACC__=double')
        elif '-D__TYPEACC__=float' in self.optional_flags:
            c_dtype_acc = 'float'
            self.optional_flags.remove('-D__TYPEACC__=float')
        else:
            raise ValueError('not implemented')
            
        if '-DSUM_SCHEME=0' in self.optional_flags:
            sum_scheme = 'direct_sum'
            self.optional_flags.remove('-DSUM_SCHEME=0')
        elif '-DSUM_SCHEME=1' in self.optional_flags:
            sum_scheme = 'block_sum'
            self.optional_flags.remove('-DSUM_SCHEME=1')
        elif '-DSUM_SCHEME=2' in self.optional_flags:
            sum_scheme = 'kahan_scheme'
            self.optional_flags.remove('-DSUM_SCHEME=2')
        else:
            print(self.optional_flags)
            raise ValueError('not implemented')
        
        if '-DENABLECHUNK=1' in self.optional_flags:
            print("[KeOps] warning : chunk mode is not yet implemented in new KeOps engine, option is deactivated.")
            self.optional_flags.remove('-DENABLECHUNK=1')
        
        if self.optional_flags:
            print("[KeOps] warning : there are options not yet implemented in new KeOps engine, these options are deactivated.")
            print("Options are:", self.optional_flags)
            
        map_reduce_id = "CpuReduc" if tagCPUGPU==0 else "GpuReduc1D" 
        
        if tag1D2D==1:
            print("[KeOps] warning : GPU_2D method is not yet implemented in new KeOps engine, switching to GPU_1D.")
        
        if tagHostDevice==0 and tagCPUGPU==1:
            raise ValueError('[KeOps] not implemented')
        
        if device["cat"] == "cpu":
            device_id = -1
        else:
            device_id = device["index"]
            
        if device_id_ != -1 and device_id_ != device_id:
            raise ValueError('[KeOps] internal error : device_id_ and device_id do not match, needs investigation...')
        
        if ranges:
            raise ValueError('[KeOps] ranges are not yet implemented in new KeOps engine')
        
        if max(list(len(arg.shape) for arg in args)) > 2:
            raise ValueError('[KeOps] reductions with batch dimensions are not yet implemented in new KeOps engine')
        
        myfun = get_keops_routine(map_reduce_id, self.red_formula_string, self.aliases, nargs, c_dtype, c_dtype_acc, sum_scheme)
        self.tagIJ = myfun.tagI
        self.dimout = myfun.dim
        M, N = (nx, ny) if myfun.tagI==0 else (ny, nx)
        out = tools.zeros((M, myfun.dim), dtype=dtype, device=device)
        out_ctype = tools.ctypes(out)
        args_ctype = (tools.ctypes(arg) for arg in args)
        myfun(M, N, device_id, out_ctype, *args_ctype)
        return out


    genred_pytorch = genred
    genred_numpy = genred   

                
        
    def import_module(self):
        return self
        
