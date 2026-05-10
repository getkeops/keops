import sys

import pykeops.config as pykeopsconfig

from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
from keopscore.utils.Cache import Cache_partial
from keopscore.utils.system_utils import KeOps_OS_Run
from keopscore.utils.messages import KeOps_Error

from pykeops.common.keops_io.LoadKeOps import LoadKeOps
from pykeops.common.utils import pyKeOps_Message


class LoadKeOps_nvrtc_class(LoadKeOps):
    def __init__(self, *args, fast_init=False):
        super().__init__(*args, fast_init=fast_init)

    def init_phase2(self):
        import importlib

        if pykeopsconfig.get_build_folder() not in sys.path:
            # The build folder is supposed to be in the python path, if not,
            # we add it
            sys.path.append(pykeopsconfig.get_build_folder())

        pykeops_nvrtc = importlib.import_module("pykeops_nvrtc")

        if self.params.c_dtype == "float":
            self.launch_keops = pykeops_nvrtc.KeOps_module_float(
                self.params.device_id_request,
                self.params.nargs,
                self.params.low_level_code_file,
            )
        elif self.params.c_dtype == "double":
            self.launch_keops = pykeops_nvrtc.KeOps_module_double(
                self.params.device_id_request,
                self.params.nargs,
                self.params.low_level_code_file,
            )
        elif self.params.c_dtype == "half2":
            self.launch_keops = pykeops_nvrtc.KeOps_module_half2(
                self.params.device_id_request,
                self.params.nargs,
                self.params.low_level_code_file,
            )

    def call_keops(self, nx, ny):
        self.launch_keops(
            self.params.tagHostDevice,
            self.params.dimy,
            nx,
            ny,
            self.params.tagI,
            self.params.tagZero,
            self.params.use_half,
            self.params.tag1D2D,
            self.params.dimred,
            self.params.cuda_block_size,
            self.params.use_chunk_mode,
            self.params.indsi,
            self.params.indsj,
            self.params.indsp,
            self.params.dim,
            self.params.dimsx,
            self.params.dimsy,
            self.params.dimsp,
            self.ranges_ptr_new,
            self.outshape,
            self.out_ptr,
            self.args_ptr_new,
            self.argshapes_new,
        )

    def import_module(self):
        return self


def compile_jit_binary():
    """
    This function compile the main .so entry point to keops_nvrt binder...
    """
    compile_command = Gpu_link_compile.get_compile_command(
        extra_flags=pykeopsconfig.python_includes,
        sourcename=pykeopsconfig.pykeops_nvrtc_name(type="src"),
        dllname=pykeopsconfig.pykeops_nvrtc_name(type="target"),
    )
    pyKeOps_Message("Compiling nvrtc binder for python", flush=True, end="", level=1)
    pyKeOps_Message(
        " in cache folder " + pykeopsconfig.path.get_build_folder(),
        flush=True,
        end="",
        level=2,
    )
    pyKeOps_Message(" ... ", flush=True, end="", use_tag=False, level=1)

    out = KeOps_OS_Run(compile_command)
    if out.returncode != 0:
        KeOps_Error(
            f"Error compiling nvrtc binder {pykeopsconfig.pykeops_nvrtc_name(type='target')} for python. See compiler output above."
        )
    else:
        pyKeOps_Message("OK", use_tag=False, flush=True, level=1)


LoadKeOps_nvrtc = Cache_partial(
    LoadKeOps_nvrtc_class,
    use_cache_file=True,
    save_folder=pykeopsconfig.get_build_folder(),
)
