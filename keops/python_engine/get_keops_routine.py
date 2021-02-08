from utils import *
from map_reduce import *


class create_or_load:
    library = {}
    @staticmethod
    def __call__(cls, *args):
        cls_id = str(cls)
        if cls_id not in create_or_load.library:
            create_or_load.library[cls_id] = {}
        cls_library = create_or_load.library[cls_id]
        hash_name = get_hash_name(*args)
        if hash_name in cls_library:
            return cls_library[hash_name]
        else:
            obj = cls(*args)
            cls_library[hash_name] = obj
            return obj
            


class get_keops_routine_class:
    
    def __init__(self, map_reduce_id, red_formula_string, nargs, dtype, *params):
        self.nargs = nargs
        self.dtype = dtype
        self.dll = None
        map_reduce_class = eval(map_reduce_id)
        self.map_reduce_obj = map_reduce_class(red_formula_string, nargs, dtype, *params)
        
        # detecting the case of formula being equal to zero, to bypass reduction.
        # This part is ugly, needs refactoring...
        if isinstance(self.map_reduce_obj.red_formula.formula, Zero):
             if isinstance(self.map_reduce_obj.red_formula, Sum_Reduction):
                 self.map_reduce_obj = map_reduce_class.AssignZero(red_formula_string, nargs, dtype, *params)
                 
        self.load_dll()
            
    def load_dll(self):
        res = self.map_reduce_obj.get_dll_and_params()
        self.dllname = res["dllname"]
        self.dll = CDLL(self.dllname)            
        self.tagI = res["tagI"]
        self.dim = res["dim"]
        ctype = eval(f"c_{self.dtype}")
        self.dll.argtypes = [c_int, c_int, c_int, POINTER(ctype)] + [POINTER(ctype)]*self.nargs
        return self
        
    def __call__(self, nx, ny, device_id, out_ptr, *args_ptr):
        if self.dll is not None:
            c_args = [c_void_p(ptr) for ptr in args_ptr]
            self.dll.Eval(c_int(nx), c_int(ny), c_int(device_id), c_void_p(out_ptr), *c_args)
   
            
def get_keops_routine(*args):
    return create_or_load()(get_keops_routine_class, *args)