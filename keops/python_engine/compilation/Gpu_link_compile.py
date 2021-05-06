from keops.python_engine.compilation import link_compile
            
class Gpu_link_compile(link_compile):
    
    source_code_extension = "cu"
    compiler = "nvcc"
    compile_options = ["-shared", "-Xcompiler -fPIC", "-O3"]
    
