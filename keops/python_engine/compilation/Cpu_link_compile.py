import os

from keops.python_engine.compilation import link_compile
from keops.python_engine.config import use_OpenMP, base_dir_path
            
class Cpu_link_compile(link_compile):

    source_code_extension = "cpp"
    
    bindings_source_dir = base_dir_path+"binders"
    
    compiler = "g++"
    compile_options = ["-shared", "-fPIC", "-O3", "-flto", "-I"+bindings_source_dir]
    
    if use_OpenMP:
        import platform
        if platform.system()=="Darwin":
            compile_options += ["-Xclang -fopenmp", "-lomp"]
            # warning : this is unsafe hack for OpenMP support on mac...
            #os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        else:
            compile_options += ["-fopenmp", "-fno-fat-lto-objects"]
    
