import os
import types
from functools import reduce

import numpy as np

import keops.config.config
import pykeops
from keops.get_keops_dll import get_keops_dll
from keops.utils.Cache import Cache_partial
from pykeops.common.parse_type import parse_dtype_acc
from pykeops.common.utils import pyKeOps_Message


class LoadKeOps_class:
    null_range = np.array([-1], dtype="int32")
    empty_ranges_new = tuple([null_range.__array_interface__["data"][0]] * 7)

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
        # TODO: the tuple length is added at the beginning. Should be remove with Pybind11
        params.indsi = (len(indsi),) + indsi
        params.indsj = (len(indsj),) + indsj
        params.indsp = (len(indsp),) + indsp
        params.dimsx = (len(dimsx),) + dimsx
        params.dimsy = (len(dimsy),) + dimsy
        params.dimsp = (len(dimsp),) + dimsp

        params.tagCPUGPU = tagCPUGPU
        params.device_id_request = device_id_request
        params.nargs = nargs

        params.reduction_op = params.red_formula_string.split("(")[0]
        params.axis = 1 - params.tagI

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
            tag = os.path.basename(params.dllname).split(".")[0]
            launch_keops_fun_name = ("launch_keops_cpu_" + tag)
            srcname = os.path.join(keops.config.config.build_path, "keops_io_cpp_" + tag + ".cpp")
            import sysconfig
            dllname = os.path.join(keops.config.config.build_path, "keops_io_cpp_" + tag + sysconfig.get_config_var('EXT_SUFFIX'))

            f = open(srcname, "w")
            f.write(
                f"""
#include "{tag}.cpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

template < typename TYPE >
int launch_keops_{tag}_cpu(int nx, int ny, int tagI, int use_half,
                     py::tuple py_ranges,
                     long py_out, int nargs,
                     py::tuple py_arg,
                     py::tuple py_argshape){{

          // Cast the ranges arrays
          std::vector< void* > ranges_v(py_ranges.size());
          for (int i = 0; i < py_ranges.size(); i++)
            ranges_v[i] = (void*) py::cast< long >(py_ranges[i]);
          //int **ranges = (int**) ranges_v.data();

        void *out_void = (void*) py_out;
        // std::cout << "out_ptr : " << (long) out << std::endl;

        std::vector< void* > arg_v(py_arg.size());
          for (int i = 0; i < py_arg.size(); i++)
            arg_v[i] = (void*) py::cast< long >(py_arg[i]);
        //TYPE **arg = (TYPE**) arg_v.data();

        std::vector< std::vector< int > > argshape_v(py_argshape.size());
        std::vector< int* > argshape_ptr_v(py_argshape.size());
        for (auto i = 0; i < py_argshape.size(); i++){{
            py::tuple tmp = py_argshape[i];
            std::vector< int > tmp_v(tmp.size());
            for (auto j =0; j < tmp.size(); j++)
                tmp_v[j] = py::cast< int >(tmp[j]);
            argshape_v[i] = tmp_v;
             argshape_ptr_v[i] = argshape_v[i].data();
        }}

        int **argshape = argshape_ptr_v.data();
        
        return {launch_keops_fun_name}(nx, ny, tagI, use_half,
                     ranges_v,
                     out_void, nargs,
                     arg_v,
                     argshape_ptr_v);


}}

PYBIND11_MODULE(keops_io_cpp_{tag}, m) {{
    m.doc() = "pyKeOps: KeOps for pytorch through pybind11 (pytorch flavour).";
    m.def("launch_keops_cpu_float", &launch_keops_{tag}_cpu < float >, "Entry point to keops - float .");
    m.def("launch_keops_cpu_double", &launch_keops_{tag}_cpu < double >, "Entry point to keops - float .");
}}                     
            """)
            f.close()

            compile_command = f"{keops.config.config.cxx_compiler} {keops.config.config.cpp_flags} {pykeops.config.python_includes} {srcname} -o {dllname}"
            os.system(compile_command)

            import importlib
            mylib = importlib.import_module("keops_io_cpp_" + tag)
            if params.c_dtype == "float":
                self.launch_keops_cpu = mylib.launch_keops_cpu_float
            if params.c_dtype == "double":
                self.launch_keops_cpu = mylib.launch_keops_cpu_double

        else:
            import keops_io_nvrtc
            if params.c_dtype == "float":
                self.launch_keops = keops_io_nvrtc.KeOps_module_float(params.device_id_request, params.nargs,
                                                                   params.low_level_code_file)
            elif params.c_dtype == "double":
                self.launch_keops = keops_io_nvrtc.KeOps_module_double(params.device_id_request, params.nargs,
                                                                    params.low_level_code_file)
            elif params.c_dtype == "half2":
                self.launch_keops = keops_io_nvrtc.KeOps_module_half2(params.device_id_request, params.nargs,
                                                                   params.low_level_code_file)

    def genred(
        self, device_args, ranges, nx, ny, nbatchdims, out, *args,
    ):

        params = self.params

        if params.use_half:
            from pykeops.torch.half2_convert import preprocess_half2

            args, ranges, tag_dummy, N = preprocess_half2(
                args, params.aliases_old, params.axis, ranges, nx, ny
            )

        # get ranges argument
        if not ranges:
            ranges_ptr_new = self.empty_ranges_new
        else:
            ranges_shapes = self.tools.array(
                [r.shape[0] for r in ranges], dtype="int32", device="cpu"
            )
            ranges = [*ranges, ranges_shapes]
            ranges_ptr_new = tuple([self.tools.get_pointer(r) for r in ranges])

        args_ptr_new = tuple([self.tools.get_pointer(arg) for arg in args])

        # get all shapes of arguments
        argshapes_new = tuple([(len(arg.shape),) + arg.shape for arg in args])

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
        out_ptr = self.tools.get_pointer(out)

        outshape = (len(out.shape),) + out.shape

        if params.tagCPUGPU == 0:
            self.launch_keops_cpu(
                nx,
                ny,
                params.tagI,
                params.use_half,
                ranges_ptr_new,
                out_ptr,
                params.nargs,
                args_ptr_new,
                argshapes_new,
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
                ranges_ptr_new,
                outshape,
                out_ptr,
                args_ptr_new,
                argshapes_new,
            )

        if params.dtype == "float16":
            from pykeops.torch.half2_convert import postprocess_half2

            out = postprocess_half2(out, tag_dummy, params.reduction_op, N)

        return out

    genred_pytorch = genred
    genred_numpy = genred

    def import_module(self):
        return self


def compile_jit_binary():
    """
    This function compile the main .so entry point to keops_nvrt binder...
    """
    from keops.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
    compile_command = Gpu_link_compile.get_compile_command(
        extra_flags=pykeops.config.python_includes,
        sourcename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "keops_io_nvrtc.cpp"),
        dllname=pykeops.config.jit_binary_name
    )
    pyKeOps_Message("Compiling nvrtc binder for python ... ", flush=True, end="")
    os.system(compile_command)
    print("OK", flush=True)


LoadKeOps = Cache_partial(
    LoadKeOps_class, use_cache_file=True, save_folder=keops.config.config.build_path
)
