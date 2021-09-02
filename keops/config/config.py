import os
from ctypes import CDLL, RTLD_GLOBAL
import keops

# System Path
base_dir_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
template_path = base_dir_path + "templates" + os.path.sep
bindings_source_dir = base_dir_path + "include" + os.path.sep
build_path = base_dir_path + "build" + os.path.sep
cuda_path = [os.path.sep + os.path.join("opt", "cuda"),
             os.path.sep + os.path.join("usr", "local", "cuda"),]

# Compiler
cxx_compiler ="g++"
compile_options = "-shared -fPIC -O3 "
use_cuda = 0

# cpp options
use_OpenMP = True # flag for OpenMP support
cpp_flags = compile_options + "-flto "
disable_pragma_unrolls = True

if use_OpenMP:
    import platform

    if platform.system() == "Darwin":
        pass
        # cpp_flags += ["-Xclang -fopenmp", "-lomp"]
        # warning : this is unsafe hack for OpenMP support on mac...
        # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    else:
        cpp_flags += "-fopenmp -fno-fat-lto-objects"

cpp_flags += " -I" + bindings_source_dir

# nvrtc options
dependencies = ["cuda", "nvrtc"]

nvrtc_flags = compile_options + "-fpermissive" + " -l" + " -l".join(dependencies)

generate_cuda_path = lambda _cuda_path: "-L" + os.path.join(_cuda_path, "lib64") \
                       + " -L" + os.path.join(_cuda_path, "targets", "x86_64-linux", "lib") \
                       + " -I" + os.path.join(_cuda_path, "targets", "x86_64-linux", "include")
nvrtc_include = " ".join([generate_cuda_path(path) for path in cuda_path]) + " -I" + bindings_source_dir

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
