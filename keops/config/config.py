import os

# System Path
base_dir_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + ".." + os.path.sep
template_path = base_dir_path + "templates" + os.path.sep
bindings_source_dir = base_dir_path + "include" + os.path.sep
build_path = base_dir_path + "build" + os.path.sep
cuda_path = [os.path.sep + os.path.join("opt", "cuda"),
             os.path.sep + os.path.join("usr", "local", "cuda"),]

# Compiler
cxx_compiler ="g++"

# nvrtc options
nvrtc_flags = " -shared -fPIC -lcuda -lnvrtc -fpermissive "

generate_cuda_path = lambda _cuda_path: "-L" + os.path.join(_cuda_path, "lib64") \
                       + " -L" + os.path.join(_cuda_path, "targets", "x86_64-linux", "lib") \
                       + " -I" + os.path.join(_cuda_path, "targets", "x86_64-linux", "include")
nvrtc_include = " ".join([generate_cuda_path(path) for path in cuda_path]) + " -I" + bindings_source_dir

jit_source_file = os.path.join(base_dir_path, "binders", "nvrtc", "keops_nvrtc.cpp")
jit_binary = os.path.join(build_path, "keops_nvrtc.so")


# flag for OpenMP support
use_OpenMP = True

