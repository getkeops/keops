import os
from keops.python_engine.compilation import link_compile
from keops.python_engine.config import build_path
from keops.python_engine import use_jit
            
class Gpu_link_compile(link_compile):
    
    source_code_extension = "cu"
    if use_jit:
        # these are used for JIT compiling mode
        low_level_code_extension = "ptx"
    else:
        # these are used for command line compiling mode
        compiler = "nvcc"
        compile_options = ["-shared", "-Xcompiler -fPIC", "-O3"]
    
    
    
