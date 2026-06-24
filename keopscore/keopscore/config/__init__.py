import os
import shutil

# Import the configuration classes
from .Cuda import CudaConfig
from .CxxCompiler import CxxCompilerConfig
from .Debug import DebugConfig
from .KeOpsPath import KeOpsPathConfig
from .OpenMP import OpenMPConfig
from .Platform import PlatformConfig
from .ReductionTuning import ReductionTuningConfig
from keopscore.utils.messages import KeOps_Error, KeOps_Message

# Instantiate the configurations once at import time to preserve the existing API.
debug = DebugConfig()
platform = PlatformConfig()
cxx = CxxCompilerConfig(platform)
openmp = OpenMPConfig(platform, cxx)
cuda = CudaConfig(platform)
path = KeOpsPathConfig(platform, cuda)
reduction = ReductionTuningConfig()


def keops_jit_compile_name(type="src"):
    basename = "nvrtc_jit"
    if type == "src":
        return os.path.join(
            path.get_base_dir_path(), "binders", "nvrtc", basename + ".cpp"
        )

    return path.get_python_extension_path(basename, suffix="SHLIB_SUFFIX")


def check_health(infos="all"):
    """
    Check the health of the specified configuration.
    """

    if infos == "all" or infos == "platform":
        platform.print_all()
    if infos == "all" or infos == "cxx":
        cxx.print_all()
    if infos == "all" or infos == "openmp":
        openmp.print_all()
    if infos == "all" or infos == "cuda":
        cuda.print_all()
    if infos == "all" or infos == "path":
        path.print_all()


def clean_keops(recompile_jit_binary=True, verbose=True):
    build_path = path.get_build_folder()
    default_build_path = path.get_default_build_path()

    if os.path.abspath(build_path) != os.path.abspath(default_build_path):
        KeOps_Error(
            f"Your build folder is set to {build_path}, which is not the default build folder. For safety reasons, the clean_keops function will not delete files in this folder. If you want to clean this folder, please do it manually."
        )

    if build_path and os.path.isdir(build_path):
        jit_binary = keops_jit_compile_name(type="target")
        # TODO: Add a safety check to prevent accidental deletion of important directories.
        for entry in os.scandir(build_path):
            if recompile_jit_binary or os.path.abspath(entry.path) != os.path.abspath(
                jit_binary
            ):
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path)
                else:
                    os.remove(entry.path)

    if verbose:
        KeOps_Message(f"{build_path} has been cleaned.")

    # Re-initialize the build folder after cleaning.
    path.set_build_folder(reset_all=True)


__all__ = [
    "platform",
    "cxx",
    "openmp",
    "cuda",
    "path",
    "debug",
    "reduction",
    "check_health",
    "clean_keops",
]
