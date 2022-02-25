import os
from os.path import join
import shutil
from ctypes import CDLL, RTLD_GLOBAL
import keopscore
from ctypes.util import find_library
from keopscore.utils.misc_utils import KeOps_Warning, KeOps_Error

# global parameters can be set here :
use_cuda = True  # use cuda if possible
use_OpenMP = True  # use OpenMP if possible

# System Path
base_dir_path = os.path.abspath(join(os.path.dirname(os.path.realpath(__file__)), ".."))
template_path = join(base_dir_path, "templates")
bindings_source_dir = join(base_dir_path)
keops_cache_folder = join(os.path.expanduser("~"), ".cache", "keops")
default_build_path = join(keops_cache_folder, "build")

# init cache folder
version_cache_file = join(keops_cache_folder, "keops_version")
if os.path.exists(version_cache_file):
    v = open(version_cache_file, encoding="utf-8")
    cache_version = v.read().rstrip()
    if cache_version != keopscore.__version__:
        v.close()
        shutil.rmtree(keops_cache_folder)
        test_init_cache = True
    else:
        v.close()
        test_init_cache = False
else:
    test_init_cache = True
if test_init_cache:
    os.makedirs(keops_cache_folder, exist_ok=True)
    v = open(version_cache_file, "w", encoding="utf-8")
    v.write(keopscore.__version__)
    v.close()


# build path setter/getter

_build_path = ""


def set_build_folder(path=None, read_save_file=False, reset_all=True):
    save_file = join(keops_cache_folder, "build_folder_location.txt")
    if not path:
        if read_save_file and os.path.isfile(save_file):
            f = open(save_file, "r")
            path = f.read()
            f.close()
        else:
            path = default_build_path
    global _build_path
    _build_path = path
    os.makedirs(path, exist_ok=True)
    f = open(save_file, "w")
    f.write(path)
    if reset_all:
        keopscore.get_keops_dll.get_keops_dll.reset(new_save_folder=_build_path)
        if keopscore.config.config.use_cuda:
            from keopscore.binders.nvrtc.Gpu_link_compile import (
                Gpu_link_compile,
                jit_compile_dll,
            )

            if not os.path.exists(jit_compile_dll()):
                Gpu_link_compile.compile_jit_compile_dll()


set_build_folder(read_save_file=True, reset_all=False)


def get_build_folder():
    return _build_path


jit_binary = join(_build_path, "keops_nvrtc.so")

# Compiler
cxx_compiler = os.getenv("CXX")
if cxx_compiler is None:
    cxx_compiler = "g++"
if shutil.which(cxx_compiler) is None:
    KeOps_Warning(
        """
    The default C++ compiler could not be found on your system.
    You need to either define the CXX environment variable or a symlink to the g++ command.
    For example if g++-8 is the command you can do
      import os
      os.environ['CXX'] = 'g++-8'
    """
    )


compile_options = " -shared -fPIC -O3 -std=c++11"


# cpp options
cpp_flags = compile_options + " -flto"
disable_pragma_unrolls = True

if use_OpenMP:
    import platform

    if platform.system() == "Darwin":
        if not os.getenv("KMP_DUPLICATE_LIB_OK") == "TRUE":
            KeOps_Warning(
                "OpenMP support is disabled on Mac by default, see the doc for enabling it."
            )
            use_OpenMP = False
        else:
            cpp_flags += " -Xclang -fopenmp -lomp "
    else:
        cpp_flags += " -fopenmp -fno-fat-lto-objects"


if platform.system() == "Darwin":
    cpp_flags += " -undefined dynamic_lookup"

cpp_flags += " -I" + bindings_source_dir


from keopscore.utils.gpu_utils import get_gpu_props

cuda_dependencies = ["cuda", "nvrtc"]
if all([find_library(lib) for lib in cuda_dependencies]):
    # N.B. calling get_gpu_props issues a warning if cuda is not available, so we do not add another warning here
    cuda_available = get_gpu_props()[0] > 0
else:
    cuda_available = False
    KeOps_Warning(
        "Cuda libraries were not detected on the system ; using cpu only mode"
    )

if not use_cuda and cuda_available:
    KeOps_Warning(
        "Cuda appears to be available on your system, but use_cuda is set to False in config.py. Using cpu only mode"
    )

if use_cuda and not cuda_available:
    use_cuda = False

if use_cuda:
    from keopscore.utils.gpu_utils import (
        libcuda_folder,
        libnvrtc_folder,
        get_cuda_include_path,
        get_cuda_version,
    )

    cuda_version = get_cuda_version()
    nvrtc_flags = (
        compile_options
        + f" -fpermissive -L{libcuda_folder} -L{libnvrtc_folder} -lcuda -lnvrtc"
    )
    nvrtc_include = " -I" + bindings_source_dir
    cuda_include_path = get_cuda_include_path()
    if cuda_include_path:
        nvrtc_include += " -I" + cuda_include_path
    jit_source_file = join(base_dir_path, "binders", "nvrtc", "keops_nvrtc.cpp")
    jit_source_header = join(base_dir_path, "binders", "nvrtc", "keops_nvrtc.h")
else:
    cuda_version = None
    libcuda_folder = None
    libnvrtc_folder = None
    nvrtc_flags = None
    nvrtc_include = None
    cuda_include_path = None
    jit_source_file = None
    jit_source_header = None
    jit_binary = None

init_cudalibs_flag = False


def init_cudalibs():
    if not keopscore.config.config.init_cudalibs_flag:
        # we load some libraries that need to be linked with KeOps code
        # This is to avoid "undefined symbols" errors.
        CDLL(find_library("nvrtc"), mode=RTLD_GLOBAL)
        CDLL(find_library("cuda"), mode=RTLD_GLOBAL)
        CDLL(find_library("cudart"), mode=RTLD_GLOBAL)
        keopscore.config.config.init_cudalibs_flag = True
