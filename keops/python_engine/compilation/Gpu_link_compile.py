import os
from keops.python_engine.compilation import link_compile
from keops.python_engine.config import build_path, get_jit_binary
            
class Gpu_link_compile(link_compile):
    
    # these are used for JIT compiling mode
    jit_binary = get_jit_binary()
    low_level_code_extension = "ptx"

    # these are used for command line compiling mode
    source_code_extension = "cu"
    compiler = "nvcc"
    compile_options = ["-shared", "-Xcompiler -fPIC", "-O3"]
    
    
    
