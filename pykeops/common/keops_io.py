from pykeops.common.get_keops_routine import get_keops_routine
from ctypes import c_int, c_void_p
from functools import reduce
import time
from keops.utils.code_gen_utils import get_hash_name
from array import array

class LoadKeOps_class:

    empty_ranges = [array("i", [-1])] * 7

    def __init__(
        self, formula, aliases, dtype, lang, optional_flags=[], include_dirs=[]
    ):
        
        start = time.time()
        
        aliases_new = []
        for k, alias in enumerate(aliases):
            alias = alias.replace(" ", "")
            if "=" in alias:
                varname, var = alias.split("=")
                if "Vi" in var:
                    cat = 0
                elif "Vj" in var:
                    cat = 1
                elif "Pm" in var:
                    cat = 2
                alias_args = var[3:-1].split(",")
                if len(alias_args) == 1:
                    ind, dim = k, eval(alias_args[0])
                elif len(alias_args) == 2:
                    ind, dim = eval(alias_args[0]), eval(alias_args[1])
                alias = f"{varname}=Var({ind},{dim},{cat})"
                aliases_new.append(alias)
        self.aliases_old = aliases
        self.aliases = aliases_new
        self.lang = lang
        self.red_formula_string = formula
        self.dtype = dtype
        
        end = time.time()
        print("keops_io init, part 1 :", end-start)
        start = time.time()
        
        self.c_dtype_acc = optional_flags["dtype_acc"]
        
        self.sum_scheme = optional_flags["sum_scheme"]

        self.enable_chunks = optional_flags["enable_chunks"]

        self.enable_final_chunks = -1
        
        self.mult_var_highdim = optional_flags["multVar_highdim"]
        
        if self.lang == "torch":
            from pykeops.torch.utils import torchtools
            self.tools = torchtools
        elif self.lang == "numpy":
            from pykeops.numpy.utils import numpytools
            self.tools = numpytools
        
        end = time.time()
        print("keops_io init, part 2 :", end-start)

    def genred(
        self,
        tagCPUGPU,
        tag1D2D,
        tagHostDevice,
        device_id_request,
        ranges,
        nx,
        ny,
        axis,
        reduction_op,
        *args,
    ):

        start = time.time()

        nargs = len(args)
        device_type, device_index = self.tools.device_type_index(args[0])
        
        end = time.time()
        print("keops_io call, part 0a :", end-start)
        start = time.time()
        
        dtype = self.tools.dtype(args[0])
        dtypename = self.tools.dtypename(dtype)
        
        end = time.time()
        print("keops_io call, part 0b :", end-start)
        start = time.time()
        
        if self.dtype not in ["auto", dtypename]:
            print(
                "[KeOps] warning : setting a dtype argument in Genred different from the input dtype of tensors is not permitted anymore, argument is ignored."
            )
        
        end = time.time()
        
        print("keops_io call, part 0c :", end-start)
        start = time.time()
        
        if dtypename == "float32":
            c_dtype = "float"
            use_half = False
        elif dtypename == "float64":
            c_dtype = "double"
            use_half = False
        elif dtypename == "float16":
            c_dtype = "half2"
            use_half = True
        else:
            raise ValueError("not implemented")
        
        end = time.time()
        print("keops_io call, part 0d :", end-start)
        start = time.time()

        if not self.c_dtype_acc:
            self.c_dtype_acc = c_dtype

        if dtypename == "float16":
            from pykeops.torch.half2_convert import preprocess_half2

            args, ranges, tag_dummy, N = preprocess_half2(
                args, self.aliases_old, axis, ranges, nx, ny
            )

        if tagCPUGPU == 0:
            map_reduce_id = "CpuReduc"
        else:
            map_reduce_id = "GpuReduc"
            map_reduce_id += "1D" if tag1D2D == 0 else "2D"

        if device_type == "cpu":
            device_id_args = -1
        else:
            device_id_args = device_index

        if (
            device_id_request != -1
            and device_id_args != -1
            and device_id_request != device_id_args
        ):
            raise ValueError("[KeOps] internal error : code needs some cleaning...")

        if device_id_request == -1:
            if device_id_args == -1:
                device_id_request = 0
            else:
                device_id_request = device_id_args

        # detect the need for using "ranges" method
        # N.B. we assume here that there is at least a cat=0 or cat=1 variable in the formula...
        nbatchdims = max(len(arg.shape) for arg in args) - 2
        if nbatchdims > 0 or ranges:
            map_reduce_id += "_ranges"

        end = time.time()
        print("keops_io call, part 1 :", end-start)
        start = time.time()
        
                
        myfun = get_keops_routine(
            map_reduce_id,
            self.red_formula_string,
            self.enable_chunks,
            self.enable_final_chunks,
            self.mult_var_highdim,
            self.aliases,
            nargs,
            c_dtype,
            self.c_dtype_acc,
            self.sum_scheme,
            tagHostDevice,
            tagCPUGPU,
            tag1D2D,
            use_half,
            device_id_request,
        )
        
        end = time.time()
        print("keops_io call, part 2 :", end-start)
        start = time.time()

        self.tagIJ = myfun.tagI
        self.dimout = myfun.dim

        # get ranges argument
        if not ranges:
            ranges = self.empty_ranges
        else:
            ranges = [*ranges, self.tools.array([r.shape[0] for r in ranges], dtype="int32")]
            
        args_ptr = [c_void_p(arg.data_ptr()) for arg in args]

        # get all shapes of arguments
        argshapes = [array("i", (len(arg.shape),) + arg.shape) for arg in args]

        # initialize output array

        M = nx if myfun.tagI == 0 else ny

        if use_half:
            M += M % 2

        if nbatchdims:
            batchdims_shapes = []
            for arg in args:
                batchdims_shapes.append(list(arg.shape[:nbatchdims]))
            tmp = reduce(
                np.maximum, batchdims_shapes
            )  # this is faster than np.max(..., axis=0)
            shapeout = tuple(tmp) + (M, myfun.dim)
        else:
            shapeout = (M, myfun.dim)

        out = self.tools.empty(shapeout, dtype=dtype, device_type=device_type, device_index=device_index)
        out_ptr = c_void_p(out.data_ptr())

        outshape = array("i", (len(out.shape),) + out.shape)
        
        end = time.time()
        print("keops_io call, part 3 :", end-start)
        start = time.time()

        # call the routine
        myfun(
            c_dtype,
            nx,
            ny,
            tagHostDevice,
            device_id_request,
            ranges,
            outshape,
            out_ptr,
            args_ptr,
            argshapes,
        )

        if dtypename == "float16":
            from pykeops.torch.half2_convert import postprocess_half2

            out = postprocess_half2(out, tag_dummy, reduction_op, N)
        
        end = time.time()
        print("keops_io call, part 4 :", end-start)

        return out

    genred_pytorch = genred
    genred_numpy = genred

    def import_module(self):
        return self


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
            res = cls_library[hash_name]
        else:
            obj = cls(*args)
            cls_library[hash_name] = obj
            res = obj
        
        return res

def LoadKeOps(*args):    
    start = time.time()
    res = create_or_load()(LoadKeOps_class, *args)
    end = time.time()
    print("LoadKeOps init:", end-start)
    return res
