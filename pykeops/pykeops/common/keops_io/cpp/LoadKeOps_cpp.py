import pykeops.config as pykeopsconfig

from keopscore.utils.Cache import Cache_partial
from keopscore.utils.system_utils import KeOps_OS_Run
from keopscore.utils.messages import KeOps_Error

from pykeops.common.keops_io.LoadKeOps import LoadKeOps
from pykeops.common.utils import pyKeOps_Message


class LoadKeOps_cpp_class(LoadKeOps):
    def __init__(self, *args, fast_init=False):
        super().__init__(*args, fast_init=fast_init)

    def init_phase2(self):
        import importlib

        if should_compile_binder():
            compile_jit_binary()

        pykeops_cpp = importlib.import_module("pykeops_cpp")
        formula_library = self.params.low_level_code_file
        if not formula_library:
            raise RuntimeError(
                "[pyKeOps] Error: empty CPU formula library path. "
                "Please clean the KeOps cache and recompile formulas."
            )

        if self.params.c_dtype == "float":
            self.launch_keops_cpu = pykeops_cpp.KeOps_module_float(formula_library)
        elif self.params.c_dtype == "double":
            self.launch_keops_cpu = pykeops_cpp.KeOps_module_double(formula_library)
        else:
            raise ValueError(
                "The cpp backend only supports float32 and float64 inputs."
            )

    def call_keops(self, nx, ny):
        self.launch_keops_cpu(
            self.params.dimy,
            nx,
            ny,
            self.params.tagI,
            self.params.tagZero,
            self.params.use_half,
            self.params.dimred,
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


def should_compile_binder():
    return pykeopsconfig.path.target_needs_update(
        pykeopsconfig.pykeops_cpp_binder_name(type="target"),
        pykeopsconfig.pykeops_cpp_binder_name(type="src"),
    )


def compile_jit_binary():
    """
    Compile the reusable pybind11 entry point for the cpp backend.
    """
    compile_command = (
        f"{pykeopsconfig.cxx.get_cxx_compiler()} "
        f"{pykeopsconfig.cxx.get_compile_options()} "
        f"{pykeopsconfig.path.get_include_options()} "
        f"{pykeopsconfig.python_includes} "
        f"{pykeopsconfig.pykeops_cpp_binder_name(type='src')} "
        f"{pykeopsconfig.cxx.get_linking_options()} "
        f"{pykeopsconfig.cxx.get_dynamic_loader_linking_options()} "
        f"-o {pykeopsconfig.pykeops_cpp_binder_name(type='target')}"
    )
    pyKeOps_Message("Compiling cpp binder for python", flush=True, end="", level=1)
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
            f"Error compiling cpp binder {pykeopsconfig.pykeops_cpp_binder_name(type='target')} for python. See compiler output above."
        )
    else:
        pyKeOps_Message("OK", use_tag=False, flush=True, level=1)


LoadKeOps_cpp = Cache_partial(
    LoadKeOps_cpp_class,
    use_cache_file=True,
    save_folder=pykeopsconfig.path.get_build_folder(),
)
