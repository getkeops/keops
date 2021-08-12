import os

base_dir_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
template_path = base_dir_path + "templates"
build_path = base_dir_path + "build" + os.path.sep
cuda_path = [os.path.sep + os.path.join("opt", "cuda"), os.path.sep + os.path.join("usr", "local", "cuda")]

# flag for OpenMP support
use_OpenMP = True

def get_jit_binary(gpu_props_compile_flags, check_compile=True):
    # Returns the path to the main KeOps binary (dll) that will be used to JIT compile all formulas.
    # If the dll is not present, it compiles it from source, except if check_compile is False.
    jit_source_file = (
        os.path.dirname(os.path.realpath(__file__))
        + os.path.sep
        + "compilation"
        + os.path.sep
        + "keops_nvrtc.cpp"
    )
    jit_binary = os.path.join(build_path, "keops_nvrtc.so")

    if check_compile and not os.path.exists(jit_binary):
        print("[KeOps] Compiling main dll...", flush=True, end="")
        f = lambda _cuda_path:  "-L" + os.path.join(_cuda_path, "lib64") + " -L" + os.path.join(_cuda_path, "targets", "x86_64-linux", "lib") + " -I" + os.path.join(_cuda_path, "targets", "x86_64-linux", "include")
        flags = " ".join([f(path) for path in cuda_path])
        flags += " -shared -fPIC -lcuda -lnvrtc -fpermissive "
        flags += gpu_props_compile_flags
        jit_compile_command = f"g++  {flags} {jit_source_file} -o {jit_binary}"
        os.system(jit_compile_command)
        print("Done.", flush=True)
    return jit_binary


def clean_keops(delete_jit_binary=False):
    from keops.python_engine import gpu_props_compile_flags
    jit_binary = get_jit_binary(gpu_props_compile_flags, check_compile=False)
    for f in os.scandir(build_path):
        if f.path != jit_binary or delete_jit_binary:
            os.remove(f.path)
    print(f"[KeOps] Folder {build_path} has been cleaned.")
