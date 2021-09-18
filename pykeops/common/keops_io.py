from pykeops.common.get_keops_routine import get_keops_routine
from ctypes import c_int, c_void_p
import numpy as np
from functools import reduce


class LoadKeOps:
    @staticmethod
    def numpy_array2ctypes(x):
        return dict(data=c_void_p(x.ctypes.data), type=c_void_p)

    @staticmethod
    def ranges2ctype(ranges, array2ctypes):
        ranges_ctype = list(array2ctypes(r) for r in ranges)
        ranges_ctype = (c_void_p * 7)(*(r["data"] for r in ranges_ctype))
        return ranges_ctype

    empty_ranges = (np.array([-1], dtype="int32"),) * 7
    empty_ranges_ctype = ranges2ctype.__func__(
        empty_ranges, numpy_array2ctypes.__func__
    )

    def __init__(
        self, formula, aliases, dtype, lang, optional_flags=[], include_dirs=[]
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
        self.aliases_old = aliases
        self.aliases = aliases_new
        self.lang = lang
        self.optional_flags = optional_flags
        self.red_formula_string = formula
        self.dtype = dtype

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

        if self.lang == "torch":
            from pykeops.torch.utils import torchtools

            tools = torchtools
        elif self.lang == "numpy":
            from pykeops.numpy.utils import numpytools

            tools = numpytools

        nargs = len(args)
        device_args = tools.device_dict(args[0])
        dtype = tools.dtype(args[0])
        dtypename = tools.dtypename(dtype)
        if self.dtype not in ["auto", dtypename]:
            print(
                "[KeOps] warning : setting a dtype argument in Genred different from the input dtype of tensors is not permitted anymore, argument is ignored."
            )

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

        if dtypename == "float16":
            from pykeops.torch.half2_convert import preprocess_half2

            args, ranges, tag_dummy, N = preprocess_half2(
                args, self.aliases_old, axis, ranges, nx, ny
            )

        if "-D__TYPEACC__=double" in self.optional_flags:
            c_dtype_acc = "double"
            self.optional_flags.remove("-D__TYPEACC__=double")
        elif "-D__TYPEACC__=float" in self.optional_flags:
            c_dtype_acc = "float"
            self.optional_flags.remove("-D__TYPEACC__=float")
        elif "-D__TYPEACC__=half2" in self.optional_flags:
            c_dtype_acc = "half2"
            self.optional_flags.remove("-D__TYPEACC__=half2")
        elif "-D__TYPEACC__=float2" in self.optional_flags:
            c_dtype_acc = "float2"
            self.optional_flags.remove("-D__TYPEACC__=float2")
        else:
            c_dtype_acc = c_dtype

        if "-DSUM_SCHEME=0" in self.optional_flags:
            sum_scheme = "direct_sum"
            self.optional_flags.remove("-DSUM_SCHEME=0")
        elif "-DSUM_SCHEME=1" in self.optional_flags:
            sum_scheme = "block_sum"
            self.optional_flags.remove("-DSUM_SCHEME=1")
        elif "-DSUM_SCHEME=2" in self.optional_flags:
            sum_scheme = "kahan_scheme"
            self.optional_flags.remove("-DSUM_SCHEME=2")
        else:
            sum_scheme = "block_sum"

        enable_chunks = -1
        if "-DENABLECHUNK=1" in self.optional_flags:
            enable_chunks = 1
            self.optional_flags.remove("-DENABLECHUNK=1")
        elif "-DENABLECHUNK=0" in self.optional_flags:
            enable_chunks = 0
            self.optional_flags.remove("-DENABLECHUNK=0")

        enable_final_chunks = -1
        if "-DENABLE_FINAL_CHUNKS=1" in self.optional_flags:
            enable_final_chunks = 1
            self.optional_flags.remove("-DENABLE_FINAL_CHUNKS=1")
        elif "-DENABLE_FINAL_CHUNKS=0" in self.optional_flags:
            enable_final_chunks = 0
            self.optional_flags.remove("-DENABLE_FINAL_CHUNKS=0")

        mult_var_highdim = -1
        if "-DMULT_VAR_HIGHDIM=1" in self.optional_flags:
            mult_var_highdim = 1
            self.optional_flags.remove("-DMULT_VAR_HIGHDIM=1")
        elif "-DMULT_VAR_HIGHDIM=0" in self.optional_flags:
            mult_var_highdim = 0
            self.optional_flags.remove("-DMULT_VAR_HIGHDIM=0")

        if self.optional_flags:
            print(
                "[KeOps] warning : there are options not yet implemented in new KeOps engine, these options are deactivated."
            )
            print("Options are:", self.optional_flags)

        if tagCPUGPU == 0:
            map_reduce_id = "CpuReduc"
        else:
            map_reduce_id = "GpuReduc"
            map_reduce_id += "1D" if tag1D2D == 0 else "2D"

        if device_args["cat"] == "cpu":
            device_id_args = -1
        else:
            device_id_args = device_args["index"]

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
        # N.B. we assume here that there is a least a cat=0 or cat=1 variable in the formula...
        nbatchdims = max(len(arg.shape) for arg in args) - 2
        if nbatchdims > 0 or ranges:
            map_reduce_id += "_ranges"

        myfun = get_keops_routine(
            map_reduce_id,
            self.red_formula_string,
            enable_chunks,
            enable_final_chunks,
            mult_var_highdim,
            self.aliases,
            nargs,
            c_dtype,
            c_dtype_acc,
            sum_scheme,
            tagHostDevice,
            tagCPUGPU,
            tag1D2D,
            use_half,
            device_id_request,
        )

        self.tagIJ = myfun.tagI
        self.dimout = myfun.dim

        # get ranges argument as ctypes
        if not ranges:
            ranges_ctype = self.empty_ranges_ctype
        else:
            ranges = (*ranges, tools.array([r.shape[0] for r in ranges], dtype="int32"))
            ranges_ctype = self.ranges2ctype(ranges, tools.ctypes)

        # convert arguments arrays to ctypes
        args_ctype = [tools.ctypes(arg) for arg in args]

        # get all shapes of arguments as ctypes
        argshapes_ctype = [
            (c_int * (len(arg.shape) + 1))(*((len(arg.shape),) + arg.shape))
            for arg in args
        ]

        # initialize output array and converting to ctypes

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

        out = tools.empty(shapeout, dtype=dtype, device=device_args)

        outshape_ctype = (c_int * (len(out.shape) + 1))(
            *((len(out.shape),) + out.shape)
        )

        out_ctype = tools.ctypes(out)

        # call the routine
        myfun(
            c_dtype,
            nx,
            ny,
            tagHostDevice,
            device_id_request,
            ranges_ctype,
            outshape_ctype,
            out_ctype,
            args_ctype,
            argshapes_ctype,
        )

        if dtypename == "float16":
            from pykeops.torch.half2_convert import postprocess_half2

            out = postprocess_half2(out, tag_dummy, reduction_op, N)

        return out

    genred_pytorch = genred
    genred_numpy = genred

    def import_module(self):
        return self
