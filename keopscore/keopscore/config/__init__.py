import os
import shutil

# Import the configuration classes
from .Cuda import CudaConfig
from .CxxCompiler import CxxCompilerConfig
from .Debug import DebugConfig
from .KeOpsPath import KeOpsPathConfig
from .OpenMP import OpenMPConfig
from .Platform import PlatformConfig
from .chunks import ChunksConfig
from keopscore.utils.messages import KeOps_Message

# Instantiate the configurations once at import time to preserve the existing API.
debug = DebugConfig()
platform = PlatformConfig()
cxx = CxxCompilerConfig(platform)
openmp = OpenMPConfig(platform, cxx)
cuda = CudaConfig()
path = KeOpsPathConfig(platform, cuda)
chunks = ChunksConfig()

# flag for automatic factorization : apply automatic factorization for all formulas before reduction.
auto_factorize = False


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
    use_cuda = cuda.get_use_cuda()
    jit_binary = path.get_jit_binary() if use_cuda else None

    if build_path and os.path.isdir(build_path):
        for entry in os.scandir(build_path):
            if recompile_jit_binary or entry.path != jit_binary:
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path)
                else:
                    os.remove(entry.path)

    if verbose:
        KeOps_Message(f"{build_path} has been cleaned.")

    from keopscore.get_keops_dll import get_keops_dll

    get_keops_dll.reset()
    if use_cuda and recompile_jit_binary:
        from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile

        Gpu_link_compile.compile_jit_compile_dll()


__all__ = [
    "platform",
    "cxx",
    "openmp",
    "cuda",
    "path",
    "debug",
    "auto_factorize",
    "check_health",
    "clean_keops",
]
