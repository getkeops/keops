import os

base_dir_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep
template_path = base_dir_path + "templates"
build_path = base_dir_path + "build" + os.path.sep

# flag for OpenMP support
use_OpenMP = True  

def get_jit_binary():
    jit_source_file = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "compilation" + os.path.sep + "keops_nvrtc.cu"
    jit_binary = build_path + os.path.sep + "keops_nvrtc.so"
    if not os.path.exists(jit_binary):
        print("[KeOps] Compiling main dll...", flush=True, end='')
        jit_compile_command = f"nvcc -shared -Xcompiler -fPIC -lnvrtc -lcuda {jit_source_file} -o {jit_binary}"
        os.system(jit_compile_command)
        print("Done.", flush=True)
    return jit_binary
