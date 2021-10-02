from pykeops.common.parse_type import parse_dtype_acc
from ctypes import c_int, c_void_p
from functools import reduce
from array import array
import time
from keops.get_keops_dll import get_keops_dll
import cppyy
import numpy as np
from keops.config.config import get_build_folder
import os
import pickle
import types


class LoadKeOps_class:

    null_range = np.array([-1], dtype="int32")
    empty_ranges = [c_void_p(null_range.__array_interface__["data"][0])] * 7

    def __init__(self, *args, fast_init=False):
        if fast_init:
            self.params = args[0]
        else:
            self.init(*args)
        self.dimout = self.params.dim
        self.tagIJ = self.params.tagI
        self.init_phase2()

    def init(
        self,
        tagCPUGPU,
        tag1D2D,
        tagHostDevice,
        use_ranges,
        device_id_request,
        formula,
        aliases,
        nargs,
        dtype,
        lang,
        optional_flags,
    ):

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

        params = types.SimpleNamespace()
        params.aliases_old = aliases
        params.aliases = aliases_new
        params.lang = lang
        params.red_formula_string = formula
        params.dtype = dtype
        dtype_acc = optional_flags["dtype_acc"]
        dtype_acc = parse_dtype_acc(dtype_acc, dtype)

        params.c_dtype_acc = dtype_acc
        params.sum_scheme = optional_flags["sum_scheme"]
        params.enable_chunks = optional_flags["enable_chunks"]
        params.enable_final_chunks = -1
        params.mult_var_highdim = optional_flags["multVar_highdim"]
        params.tagHostDevice = tagHostDevice

        if dtype == "float32":
            params.c_dtype = "float"
            params.use_half = False
        elif dtype == "float64":
            params.c_dtype = "double"
            params.use_half = False
        elif dtype == "float16":
            params.c_dtype = "half2"
            params.use_half = True
        else:
            raise ValueError("not implemented")

        if not params.c_dtype_acc:
            params.c_dtype_acc = params.c_dtype

        if tagCPUGPU == 0:
            map_reduce_id = "CpuReduc"
        else:
            map_reduce_id = "GpuReduc"
            map_reduce_id += "1D" if tag1D2D == 0 else "2D"

        if use_ranges:
            map_reduce_id += "_ranges"

        (
            params.dllname,
            params.low_level_code_file,
            params.tagI,
            params.tagZero,
            params.use_half,
            params.cuda_block_size,
            params.use_chunk_mode,
            params.tag1D2D,
            params.dimred,
            params.dim,
            params.dimy,
            indsi,
            indsj,
            indsp,
            dimsx,
            dimsy,
            dimsp,
        ) = get_keops_dll(
            map_reduce_id,
            params.red_formula_string,
            params.enable_chunks,
            params.enable_final_chunks,
            params.mult_var_highdim,
            params.aliases,
            nargs,
            params.c_dtype,
            params.c_dtype_acc,
            params.sum_scheme,
            params.tagHostDevice,
            tagCPUGPU,
            tag1D2D,
            params.use_half,
            device_id_request,
        )

        # now we switch indsi, indsj and dimsx, dimsy in case tagI=1.
        # This is to be consistent with the convention used in the old
        # bindings (see functions GetIndsI, GetIndsJ, GetDimsX, GetDimsY
        # from file binder_interface.h. Clearly we could do better if we
        # carefully rewrite some parts of the code
        if params.tagI == 1:
            indsi, indsj = indsj, indsi
            dimsx, dimsy = dimsy, dimsx

        params.indsi = array("i", (len(indsi),) + indsi)
        params.indsj = array("i", (len(indsj),) + indsj)
        params.indsp = array("i", (len(indsp),) + indsp)
        params.dimsx = array("i", (len(dimsx),) + dimsx)
        params.dimsy = array("i", (len(dimsy),) + dimsy)
        params.dimsp = array("i", (len(dimsp),) + dimsp)

        params.tagCPUGPU = tagCPUGPU
        params.device_id_request = device_id_request
        params.nargs = nargs

        self.params = params

    def init_phase2(self):
        params = self.params
        if params.lang == "torch":
            from pykeops.torch.utils import torchtools

            self.tools = torchtools
        elif params.lang == "numpy":
            from pykeops.numpy.utils import numpytools

            self.tools = numpytools

        if params.tagCPUGPU == 0:
            cppyy.load_library(params.dllname)
            launch_keops_fun_name = (
                "launch_keops_cpu_" + os.path.basename(params.dllname).split(".")[0]
            )
            cppyy.cppdef(
                f"""
                           int {launch_keops_fun_name}(int nx, int ny, int tagI, int use_half,
                                                     const std::vector<void*>& ranges_v,
                                                     void *out_void, int nargs, 
                                                     const std::vector<void*>& arg_v,
                                                     const std::vector<int*>& argshape_v);                         
            """
            )
            self.launch_keops_cpu = getattr(cppyy.gbl, launch_keops_fun_name)
        else:
            self.launch_keops = cppyy.gbl.KeOps_module[params.c_dtype](
                params.device_id_request, params.nargs, params.low_level_code_file
            )

    def genred(
        self, device_args, ranges, nx, ny, nbatchdims, axis, reduction_op, out, *args,
    ):

        params = self.params

        if params.use_half:
            from pykeops.torch.half2_convert import preprocess_half2

            args, ranges, tag_dummy, N = preprocess_half2(
                args, params.aliases_old, axis, ranges, nx, ny
            )

        # get ranges argument
        if not ranges:
            ranges_ptr = self.empty_ranges
        else:
            ranges_shapes = self.tools.array(
                [r.shape[0] for r in ranges], dtype="int32", device="cpu"
            )
            ranges = [*ranges, ranges_shapes]
            ranges_ptr = [c_void_p(self.tools.get_pointer(r)) for r in ranges]

        args_ptr = [c_void_p(self.tools.get_pointer(arg)) for arg in args]

        # get all shapes of arguments
        argshapes = [array("i", (len(arg.shape),) + arg.shape) for arg in args]

        # initialize output array

        M = nx if params.tagI == 0 else ny

        if params.use_half:
            M += M % 2

        if nbatchdims:
            batchdims_shapes = []
            for arg in args:
                batchdims_shapes.append(list(arg.shape[:nbatchdims]))
            tmp = reduce(
                np.maximum, batchdims_shapes
            )  # this is faster than np.max(..., axis=0)
            shapeout = tuple(tmp) + (M, params.dim)
        else:
            shapeout = (M, params.dim)

        if out is None:
            out = self.tools.empty(shapeout, dtype=args[0].dtype, device=device_args)
        out_ptr = c_void_p(self.tools.get_pointer(out))

        outshape = array("i", (len(out.shape),) + out.shape)

        if params.tagCPUGPU == 0:
            self.launch_keops_cpu(
                nx,
                ny,
                params.tagI,
                params.use_half,
                ranges_ptr,
                out_ptr,
                params.nargs,
                args_ptr,
                argshapes,
            )
        else:
            self.launch_keops(
                params.tagHostDevice,
                params.dimy,
                nx,
                ny,
                params.tagI,
                params.tagZero,
                params.use_half,
                params.tag1D2D,
                params.dimred,
                params.cuda_block_size,
                params.use_chunk_mode,
                params.indsi,
                params.indsj,
                params.indsp,
                params.dim,
                params.dimsx,
                params.dimsy,
                params.dimsp,
                ranges_ptr,
                outshape,
                out_ptr,
                args_ptr,
                argshapes,
            )

        if params.dtype == "float16":
            from pykeops.torch.half2_convert import postprocess_half2

            out = postprocess_half2(out, tag_dummy, reduction_op, N)

        return out

    genred_pytorch = genred
    genred_numpy = genred

    def import_module(self):
        return self


class library:
    def __init__(self, cls, use_cache_file=False, save_folder="."):
        self.cls = cls
        self.library = {}
        self.use_cache_file = use_cache_file
        if self.use_cache_file:
            self.cache_file = os.path.join(save_folder, cls.__name__ + "_cache.pkl")
            if os.path.isfile(self.cache_file):
                f = open(self.cache_file, "rb")
                self.library_params = pickle.load(f)
                f.close()
            else:
                self.library_params = {}
            import atexit

            atexit.register(self.save_cache)

    def __call__(self, *args):
        str_id = "".join(list(str(arg) for arg in args))
        if not str_id in self.library:
            if self.use_cache_file:
                if str_id in self.library_params:
                    params = self.library_params[str_id]
                    self.library[str_id] = self.cls(params, fast_init=True)
                else:
                    obj = self.cls(*args)
                    self.library_params[str_id] = obj.params
                    self.library[str_id] = obj
            else:
                self.library[str_id] = self.cls(*args)
        return self.library[str_id]

    def reset(self):
        self.library = {}
        if self.use_cache_file:
            self.library_params = {}

    def save_cache(self):
        f = open(self.cache_file, "wb")
        pickle.dump(self.library_params, f)
        f.close()


LoadKeOps = library(
    LoadKeOps_class, use_cache_file=True, save_folder=get_build_folder()
)
