from utils import *
from map_reduce import *


class get_keops_routine:
    
    library = {}
    
    def __init__(self, map_reduce_id, red_formula_string, nargs, dtype, *params):
        self.nargs = nargs
        self.dtype = dtype
        self.gencode_filename = get_hash_name(map_reduce_id, red_formula_string, nargs, dtype, *params)
        self.dll = None
        if self.gencode_filename in get_keops_routine.library:
            rec = get_keops_routine.library[self.gencode_filename]  
            self.dll = rec["dll"]
            self.tagI = rec["tagI"]
            self.dim = rec["dim"]
        else:
            map_reduce_class = eval(map_reduce_id)
            link_compile_class = map_reduce_class.link_compile_class
            map_reduce_obj = map_reduce_class(red_formula_string, nargs, dtype, *params)
            self.link_compile_obj = link_compile_class(map_reduce_obj, red_formula_string, nargs, dtype, *params)
            self.load_dll()
            
    def load_dll(self):
        res = self.link_compile_obj.get_dll_and_params()
        self.dllname = res["dllname"]
        if self.dllname is not "zero":
            self.dll = CDLL(self.dllname)            
        self.tagI = res["tagI"]
        self.dim = res["dim"]
        ctype = eval(f"c_{self.dtype}")
        self.dll.argtypes = [c_int, c_int, POINTER(ctype)] + [POINTER(ctype)]*self.nargs
        get_keops_routine.library[self.gencode_filename] = { "dll":self.dll, "tagI":self.tagI, "dim":self.dim }
        return self
        
    def __call__(self, nx, ny, out_ptr, *args_ptr):
        if self.dll is not None:
            c_args = [c_void_p(ptr) for ptr in args_ptr]
            self.dll.Eval(c_int(nx), c_int(ny), c_void_p(out_ptr), *c_args)