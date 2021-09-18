import os
from ctypes import CDLL, RTLD_GLOBAL
import keops
from ctypes.util import find_library
from keops.utils.misc_utils import KeOps_Warning

# System Path
base_dir_path = (
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
)


# global parameters can be set here :
use_cuda = True                                         # use cuda if possible
use_OpenMP = True                                       # use OpenMP if possible
build_path = base_dir_path + "build" + os.path.sep      # location of all JIT generated files



# System Path, continued
template_path = base_dir_path + "templates" + os.path.sep
bindings_source_dir = base_dir_path + "include" + os.path.sep


# Compiler
cxx_compiler = "g++"
compile_options = " -shared -fPIC -O3"


# cpp options
cpp_flags = compile_options + " -flto"
disable_pragma_unrolls = True

if use_OpenMP:
    import platform

    if platform.system() == "Darwin":
        use_OpenMP = False  # disabled currently, because hack below is unsafe..
        # cpp_flags += " -Xclang -fopenmp -lomp "
        # # warning : this is unsafe hack for OpenMP support on mac...
        # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    else:
        cpp_flags += " -fopenmp -fno-fat-lto-objects"

cpp_flags += " -I" + bindings_source_dir




cuda_dependencies = ["cuda", "nvrtc"]
cuda_available = all([find_library(lib) for lib in cuda_dependencies])

if not use_cuda and cuda_available:
    KeOps_Warning("Cuda appears to be available on your system, but use_cuda is set to False in config.py. Using cpu only mode")
    
if use_cuda and not cuda_available:
    KeOps_Warning("Cuda was not detected on the system ; using cpu only mode")
    use_cuda = False

# path to Cuda : currently we just list a few possible paths this is not good at all...
cuda_path = [
    os.path.sep + os.path.join("opt", "cuda"),  # for oban
    os.path.sep + os.path.join("usr", "local", "cuda"),  # for bartlett
    os.path.sep + os.path.join("usr", "local", "cuda-11.3"),  # for topdyn
]

nvrtc_flags = compile_options + " -fpermissive" + " -l" + " -l".join(cuda_dependencies)

generate_cuda_path = (
    lambda _cuda_path: "-L"
    + os.path.join(_cuda_path, "lib64")
    + " -L"
    + os.path.join(_cuda_path, "targets", "x86_64-linux", "lib")
    + " -I"
    + os.path.join(_cuda_path, "targets", "x86_64-linux", "include")
)
nvrtc_include = (
    " ".join([generate_cuda_path(path) for path in cuda_path])
    + " -I"
    + bindings_source_dir
)

jit_source_file = os.path.join(base_dir_path, "binders", "nvrtc", "keops_nvrtc.cpp")
jit_binary = os.path.join(build_path, "keops_nvrtc.so")

init_cudalibs_flag = False

def init_cudalibs():
    if not keops.config.config.init_cudalibs_flag:
        # we load some libraries that need to be linked with KeOps code
        # This is to avoid "undefined symbols" errors.
        CDLL("libnvrtc.so", mode=RTLD_GLOBAL)
        CDLL("libcuda.so", mode=RTLD_GLOBAL)
        CDLL("libcudart.so", mode=RTLD_GLOBAL)
        keops.config.config.init_cudalibs_flag = True

