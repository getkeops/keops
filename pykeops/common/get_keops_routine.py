from ctypes import create_string_buffer, c_char_p, c_int, CDLL, POINTER, c_void_p

from keops.utils.code_gen_utils import get_hash_name
from keops.get_keops_dll import get_keops_dll
import time


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
        
        start = time.time()

        (
            self.dllname,
            self.low_level_code_file,
            self.tagI,
            self.tagZero,
            self.use_half,
            self.cuda_block_size,
            self.use_chunk_mode,
            self.tag1D2D,
            self.dimred,
            self.dim,
            self.dimy,
            indsi,
            indsj,
            indsp,
            dimsx,
            dimsy,
            dimsp,
        ) = get_keops_dll(map_reduce_id, *args)

        end = time.time()
        print("total time for get_keops_dll : ", end-start)

        # now we switch indsi, indsj and dimsx, dimsy in case tagI=1.
        # This is to be consistent with the convention used in the old
        # bindings (see functions GetIndsI, GetIndsJ, GetDimsX, GetDimsY
        # from file binder_interface.h. Clearly we could do better if we
        # carefully rewrite some parts of the code
        if self.tagI == 1:
            indsi, indsj = indsj, indsi
            dimsx, dimsy = dimsy, dimsx

        self.dll = CDLL(self.dllname)
        self.indsi_ctype = (c_int * (len(indsi) + 1))(*((len(indsi),) + indsi))
        self.indsj_ctype = (c_int * (len(indsj) + 1))(*((len(indsj),) + indsj))
        self.indsp_ctype = (c_int * (len(indsp) + 1))(*((len(indsp),) + indsp))
        self.dimsx_ctype = (c_int * (len(dimsx) + 1))(*((len(dimsx),) + dimsx))
        self.dimsy_ctype = (c_int * (len(dimsy) + 1))(*((len(dimsy),) + dimsy))
        self.dimsp_ctype = (c_int * (len(dimsp) + 1))(*((len(dimsp),) + dimsp))

    def __call__(
        self,
        c_dtype,
        nx,
        ny,
        tagHostDevice,
        device_id,
        ranges_ctype,
        outshape_ctype,
        out_ctype,
        args_ctype,
        argshapes_ctype,
    ):
        start = time.time()
        
        c_args = [arg["data"] for arg in args_ctype]
        nargs = len(args_ctype)
        if c_dtype == "float":
            launch_keops = self.dll.launch_keops_float
        elif c_dtype == "double":
            launch_keops = self.dll.launch_keops_double
        elif c_dtype == "half2":
            launch_keops = self.dll.launch_keops_half
        else:
            raise ValueError("dtype", c_dtype, "not yet implemented in new KeOps engine")
        launch_keops.argtypes = (
            [
                c_char_p,  # ptx_file_name
                c_int,  # tagHostDevice
                c_int,  # dimY
                c_int,  # nx
                c_int,  # ny
                c_int,  # device_id
                c_int,  # tagI
                c_int,  # tagZero
                c_int,  # use_half
                c_int,  # tag1D2D
                c_int,  # dimred
                c_int,  # cuda_block_size
                c_int,  # use_chunk_mode
                c_int * len(self.indsi_ctype),  # indsi
                c_int * len(self.indsj_ctype),  # indsj
                c_int * len(self.indsp_ctype),  # indsp
                c_int,  # dimout
                c_int * len(self.dimsx_ctype),  # dimsx
                c_int * len(self.dimsy_ctype),  # dimsy
                c_int * len(self.dimsp_ctype),  # dimsp
                POINTER(c_void_p),  # ranges
                c_int * len(outshape_ctype),  # shapeout
                out_ctype["type"],  # out
                c_int,  # nargs
            ]
            + [arg["type"] for arg in args_ctype]  # arg
            + [c_int * len(argshape) for argshape in argshapes_ctype]  # argshape
        )

        end = time.time()
        print("time for get_keops_routine_class call, part 1 : ", end-start)

        start = time.time()
        launch_keops(
            create_string_buffer(self.low_level_code_file),
            c_int(tagHostDevice),
            c_int(self.dimy),
            c_int(nx),
            c_int(ny),
            c_int(device_id),
            c_int(self.tagI),
            c_int(self.tagZero),
            c_int(self.use_half),
            c_int(self.tag1D2D),
            c_int(self.dimred),
            c_int(self.cuda_block_size),
            c_int(self.use_chunk_mode),
            self.indsi_ctype,
            self.indsj_ctype,
            self.indsp_ctype,
            c_int(self.dim),
            self.dimsx_ctype,
            self.dimsy_ctype,
            self.dimsp_ctype,
            ranges_ctype,
            outshape_ctype,
            out_ctype["data"],
            c_int(nargs),
            *c_args,
            *argshapes_ctype
        )
        
        end = time.time()
        print("time for launch_keops call : ", end-start)


def get_keops_routine(*args):
    return create_or_load()(get_keops_routine_class, *args)
