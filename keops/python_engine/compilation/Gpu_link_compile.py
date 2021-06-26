import os
from keops.python_engine.compilation import link_compile
from keops.python_engine.config import build_path

class Gpu_link_compile(link_compile):

    source_code_extension = "cu"
    
    low_level_code_extension = "ptx"
    
    # these were used for command line compiling mode
    #compiler = "nvcc"
    #compile_options = ["-shared", "-Xcompiler -fPIC", "-O3"]
