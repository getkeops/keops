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


from keops.utils.misc_utils import find_library_abspath

libcuda_path = find_library_abspath("cuda").decode("utf-8") 
libnvrtc_path = find_library_abspath("nvrtc").decode("utf-8") 

print()
print("libcuda_path=", libcuda_path)
print("libnvrtc_path=", libnvrtc_path)
print()

nvrtc_flags = compile_options + f" -fpermissive -l{libcuda_path} -l{libnvrtc_path}"

"""
cuda_path = [
    os.path.sep + os.path.join("opt", "cuda"),  # for oban
    os.path.sep + os.path.join("usr", "local", "cuda"),  # for bartlett
    os.path.sep + os.path.join("usr", "local", "cuda-11.3"),  # for topdyn
]


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
"""

nvrtc_include = " -I" + bindings_source_dir

# trying to auto detect location of cuda headers
cuda_include_path = None
for libpath in libcuda_path, libnvrtc_path:
    for libtag in "lib", "lib64":
        libtag = os.path.sep + libtag + os.path.sep
        if libtag in libpath:
            includetag = os.path.sep + "include" + os.path.sep
            includepath = libpath.replace(libtag,includetag) + os.path.sep
            print("1***", includepath)
            if os.path.isfile(includepath + "cuda.h") and os.path.isfile(includepath + "nvrtc.h"):
                cuda_include_path = includepath
                break
            else:
                continue
        break

# if not successfull, we try a few standard locations:
if not cuda_include_path:
    from keops.utils.gpu_utils import get_cuda_version
    cuda_version = get_cuda_version()
    s = os.path.sep
    cuda_paths_to_try_start = [f"{s}opt{s}cuda{s}",
                        f"{s}usr{s}local{s}cuda{s}",
                        f"{s}usr{s}local{s}cuda-{cuda_version}{s}",
                        ]
    cuda_paths_to_try_end = [f"include{s}",
                        f"targets{s}x86_64-linux{s}include{s}",
                        ]
    for path_start in cuda_paths_to_try_start:
        for path_end in cuda_paths_to_try_end:
            path = path_start + path_end
            print("2***", path)
            if os.path.isfile(includepath + "cuda.h") and os.path.isfile(includepath + "nvrtc.h"):
                print("ok!!!!")
                cuda_include_path = includepath
                break
            else:
                print("not ok!!!!")
                continue
            print("hi guy")
            break
        
print("cuda_include_path=", cuda_include_path)

if cuda_include_path:
    nvrtc_include += " -I" + cuda_include_path

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

