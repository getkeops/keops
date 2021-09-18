import os
from ctypes import CDLL, RTLD_GLOBAL
import keops
from ctypes.util import find_library
from keops.utils.misc_utils import KeOps_Warning, KeOps_Error

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

if use_cuda:
    from keops.utils.gpu_utils import libcuda_folder, libnvrtc_folder, get_cuda_include_path
    nvrtc_flags = compile_options + f" -fpermissive -L {libcuda_folder} -L {libnvrtc_folder} -lcuda -lnvrtc"
    nvrtc_include = " -I" + bindings_source_dir
    cuda_include_path = get_cuda_include_path()
    if cuda_include_path:
        nvrtc_include += " -I" + cuda_include_path
    jit_source_file = os.path.join(base_dir_path, "binders", "nvrtc", "keops_nvrtc.cpp")
    jit_binary = os.path.join(build_path, "keops_nvrtc.so")
else:
    libcuda_folder = None
    libnvrtc_folder = None
    nvrtc_flags = None
    nvrtc_include = None
    cuda_include_path = None
    jit_source_file = None
    jit_binary = None

init_cudalibs_flag = False

def init_cudalibs():
    if not keops.config.config.init_cudalibs_flag:
        # we load some libraries that need to be linked with KeOps code
        # This is to avoid "undefined symbols" errors.
        CDLL(find_library("nvrtc"), mode=RTLD_GLOBAL)
        CDLL(find_library("cuda"), mode=RTLD_GLOBAL)
        CDLL(find_library("cudart"), mode=RTLD_GLOBAL)
        keops.config.config.init_cudalibs_flag = True

