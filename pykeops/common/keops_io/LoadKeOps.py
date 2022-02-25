import types
from functools import reduce

import numpy as np

from keopscore.get_keops_dll import get_keops_dll
from pykeops.common.parse_type import parse_dtype_acc


class LoadKeOps:
    null_range = np.array([-1], dtype="int32")
    empty_ranges_new = tuple([null_range.__array_interface__["data"][0]] * 7)

    def __init__(self, *args, fast_init):
        if fast_init:
            self.params = args[0]
        else:
            self.init(*args)
        self.dimout = self.params.dim
        self.tagIJ = self.params.tagI

        if self.params.lang == "torch":
            from pykeops.torch.utils import torchtools

            self.tools = torchtools
        elif self.params.lang == "numpy":
            from pykeops.numpy.utils import numpytools

            self.tools = numpytools

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

        self.params = types.SimpleNamespace()
        self.params.aliases_old = aliases
        self.params.aliases = aliases_new
        self.params.lang = lang
        self.params.red_formula_string = formula
        self.params.dtype = dtype
        dtype_acc = optional_flags["dtype_acc"]
        dtype_acc = parse_dtype_acc(dtype_acc, dtype)

        self.params.c_dtype_acc = dtype_acc
        self.params.sum_scheme = optional_flags["sum_scheme"]
        self.params.enable_chunks = optional_flags["enable_chunks"]
        self.params.enable_final_chunks = -1
        self.params.mult_var_highdim = optional_flags["multVar_highdim"]
        self.params.tagHostDevice = tagHostDevice

        if dtype == "float32":
            self.params.c_dtype = "float"
            self.params.use_half = False
        elif dtype == "float64":
            self.params.c_dtype = "double"
            self.params.use_half = False
        elif dtype == "float16":
            self.params.c_dtype = "half2"
            self.params.use_half = True
        else:
            raise ValueError("not implemented")

        if not self.params.c_dtype_acc:
            self.params.c_dtype_acc = self.params.c_dtype

        if tagCPUGPU == 0:
            map_reduce_id = "CpuReduc"
        else:
            map_reduce_id = "GpuReduc"
            map_reduce_id += "1D" if tag1D2D == 0 else "2D"

        if use_ranges:
            map_reduce_id += "_ranges"

        (
            self.params.tag,
            self.params.source_name,
            self.params.low_level_code_file,
            self.params.tagI,
            self.params.tagZero,
            self.params.use_half,
            self.params.cuda_block_size,
            self.params.use_chunk_mode,
            self.params.tag1D2D,
            self.params.dimred,
            self.params.dim,
            self.params.dimy,
            indsi,
            indsj,
            indsp,
            dimsx,
            dimsy,
            dimsp,
        ) = get_keops_dll(
            map_reduce_id,
            self.params.red_formula_string,
            self.params.enable_chunks,
            self.params.enable_final_chunks,
            self.params.mult_var_highdim,
            self.params.aliases,
            nargs,
            self.params.c_dtype,
            self.params.c_dtype_acc,
            self.params.sum_scheme,
            self.params.tagHostDevice,
            tagCPUGPU,
            tag1D2D,
            self.params.use_half,
            device_id_request,
        )

        # now we switch indsi, indsj and dimsx, dimsy in case tagI=1.
        # This is to be consistent with the convention used in the old
        # bindings (see functions GetIndsI, GetIndsJ, GetDimsX, GetDimsY
        # from file binder_interface.h. Clearly we could do better if we
        # carefully rewrite some parts of the code
        if self.params.tagI == 1:
            indsi, indsj = indsj, indsi
            dimsx, dimsy = dimsy, dimsx

        self.params.indsi = indsi
        self.params.indsj = indsj
        self.params.indsp = indsp
        self.params.dimsx = dimsx
        self.params.dimsy = dimsy
        self.params.dimsp = dimsp

        self.params.tagCPUGPU = tagCPUGPU
        self.params.device_id_request = device_id_request
        self.params.nargs = nargs

        self.params.reduction_op = self.params.red_formula_string.split("(")[0]
        self.params.axis = 1 - self.params.tagI

        self.init_phase1()

    def init_phase1(self):
        pass

    def init_phase2(self):
        pass

    def genred(
        self,
        device_args,
        ranges,
        nx,
        ny,
        nbatchdims,
        out,
        *args,
    ):

        if self.params.use_half:
            from pykeops.torch.half2_convert import preprocess_half2

            args, ranges, tag_dummy, N = preprocess_half2(
                args, self.params.aliases_old, self.params.axis, ranges, nx, ny
            )

        # get ranges argument
        if not ranges:
            self.ranges_ptr_new = self.empty_ranges_new
        else:
            ranges_shapes = self.tools.array(
                [r.shape[0] for r in ranges], dtype="int32", device="cpu"
            )
            ranges = [*ranges, ranges_shapes]
            self.ranges_ptr_new = tuple([self.tools.get_pointer(r) for r in ranges])

        self.args_ptr_new = tuple([self.tools.get_pointer(arg) for arg in args])

        # get all shapes of arguments
        self.argshapes_new = tuple([arg.shape for arg in args])

        # initialize output array

        M = nx if self.params.tagI == 0 else ny

        if self.params.use_half:
            M += M % 2

        if nbatchdims:
            batchdims_shapes = []
            for arg in args:
                batchdims_shapes.append(list(arg.shape[:nbatchdims]))
            tmp = reduce(
                np.maximum, batchdims_shapes
            )  # this is faster than np.max(..., axis=0)
            shapeout = tuple(tmp) + (M, self.params.dim)
        else:
            shapeout = (M, self.params.dim)

        if out is None:
            out = self.tools.empty(shapeout, dtype=args[0].dtype, device=device_args)
        self.out_ptr = self.tools.get_pointer(out)

        self.outshape = out.shape

        self.call_keops(nx, ny)

        if self.params.dtype == "float16":
            from pykeops.torch.half2_convert import postprocess_half2

            out = postprocess_half2(out, tag_dummy, self.params.reduction_op, N)

        return out

    genred_pytorch = genred
    genred_numpy = genred

    def call_keops(self):
        pass

    def import_module(self):
        return self
