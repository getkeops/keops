from ctypes import create_string_buffer, c_char_p, c_int, CDLL, POINTER, c_void_p

from keops.python_engine.utils.code_gen_utils import get_hash_name
from keops.python_engine import use_jit
from keops.python_engine.get_keops_dll import get_keops_dll


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
        (
            self.dllname,
            self.low_level_code_file,
            self.tagI,
            self.dim,
            self.dimy,
        ) = get_keops_dll(map_reduce_id, *args)
        self.dll = CDLL(self.dllname)

    def __call__(
        self, nx, ny, device_id, ranges_ctype, out_ctype, args_ctype, argshapes_ctype
    ):
        c_args = [arg["data"] for arg in args_ctype]
        nargs = len(args_ctype)
        self.dll.launch_keops.argtypes = (
            [
                c_char_p,
                c_int,
                c_int,
                c_int,
                c_int,
                c_int,
                POINTER(c_void_p),
                out_ctype["type"],
                c_int,
            ]
            + [arg["type"] for arg in args_ctype]
            + [c_int * len(argshape) for argshape in argshapes_ctype]
        )
        self.dll.launch_keops(
            create_string_buffer(self.low_level_code_file),
            c_int(self.dimy),
            c_int(nx),
            c_int(ny),
            c_int(device_id),
            c_int(self.tagI),
            ranges_ctype,
            out_ctype["data"],
            c_int(nargs),
            *c_args,
            *argshapes_ctype
        )


def get_keops_routine(*args):
    return create_or_load()(get_keops_routine_class, *args)
