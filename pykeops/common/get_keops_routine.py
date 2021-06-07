from ctypes import create_string_buffer, c_char_p, c_int, CDLL

from keops.python_engine.utils.code_gen_utils import get_hash_name
from keops.python_engine import get_keops_dll, use_jit



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
    
    def __init__(self, map_reduce_id, *args):
        self.dll = None
        if use_jit:
            self.dllname, self.low_level_code_file, self.tagI, self.dim, self.dimy = get_keops_dll(True, map_reduce_id, *args)
        else:
            self.dllname, self.tagI, self.dim = get_keops_dll(False, map_reduce_id, *args)
        self.dll = CDLL(self.dllname)            
        
    def __call__(self, nx, ny, device_id, ranges_ctype, out_ctype, *args_ctype):
        if use_jit:
            self.dll.Eval.argtypes = [c_char_p, c_int, c_int, c_int, out_ctype["type"]] + [arg["type"] for arg in args_ctype]
            c_args = [arg["data"] for arg in args_ctype]
            self.dll.Eval(create_string_buffer(self.low_level_code_file), c_int(self.dimy), c_int(nx), c_int(ny), out_ctype["data"], *c_args)
        else:
            self.dll.launch_keops.argtypes = [c_int, c_int, c_int, ranges_ctype["type"], out_ctype["type"]] + [arg["type"] for arg in args_ctype]
            c_args = [arg["data"] for arg in args_ctype]
            self.dll.launch_keops(c_int(nx), c_int(ny), c_int(device_id), ranges_ctype["data"], out_ctype["data"], *c_args)   
   
            
def get_keops_routine(*args):
    return create_or_load()(get_keops_routine_class, *args)
