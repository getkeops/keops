import os
from keops.python_engine.compilation import link_compile
            
class Gpu_link_compile(link_compile):
    
    # these are used for command line compiling mode
    source_code_extension = "cu"
    compiler = "nvcc"
    compile_options = ["-shared", "-Xcompiler -fPIC", "-O3"]
    
    # these are used for JIT compiling mode
    keops_binary = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "test_nvrtc.so"
    low_level_code_extension = "ptx"
    
