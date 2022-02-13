import os

import keops.config.config
import pykeops
from keops.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
from keops.utils.Cache import Cache_partial
from pykeops.common.keops_io.LoadKeOps import LoadKeOps
from pykeops.common.utils import pyKeOps_Message


class LoadKeOps_nvrtc_class(LoadKeOps):

    def __init__(self, *args, fast_init=False):
        super().__init__(*args, fast_init=fast_init)

    def init_phase2(self):
        import keops_io_nvrtc
        if self.params.c_dtype == "float":
            self.launch_keops = keops_io_nvrtc.KeOps_module_float(self.params.device_id_request, self.params.nargs,
                                                                  self.params.low_level_code_file)
        elif self.params.c_dtype == "double":
            self.launch_keops = keops_io_nvrtc.KeOps_module_double(self.params.device_id_request, self.params.nargs,
                                                                   self.params.low_level_code_file)
        elif self.params.c_dtype == "half2":
            self.launch_keops = keops_io_nvrtc.KeOps_module_half2(self.params.device_id_request, self.params.nargs,
                                                                  self.params.low_level_code_file)

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
        extra_flags=pykeops.config.python_includes,
        sourcename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../keops_io_nvrtc.cpp"),
        dllname=pykeops.config.jit_binary_name
    )
    pyKeOps_Message("Compiling nvrtc binder for python ... ", flush=True, end="")
    os.system(compile_command)
    print("OK", flush=True)


LoadKeOps_nvrtc = Cache_partial(
    LoadKeOps_nvrtc_class, use_cache_file=True, save_folder=keops.config.config.build_path
)
