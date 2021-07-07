import os

from keops.python_engine.compilation import link_compile
from keops.python_engine.config import use_OpenMP, base_dir_path


class Cpu_link_compile(link_compile):

    source_code_extension = "cpp"

    bindings_source_dir = base_dir_path + "binders"

    compiler = "g++"
    compile_options = ["-shared", "-fPIC", "-O3", "-flto", "-I" + bindings_source_dir]

    if use_OpenMP:
        import platform

        if platform.system() == "Darwin":
            pass
            # compile_options += ["-Xclang -fopenmp", "-lomp"]
            # warning : this is unsafe hack for OpenMP support on mac...
            # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        else:
            compile_options += ["-fopenmp", "-fno-fat-lto-objects"]

    def __init__(self):
        link_compile.__init__(self)        
        # these are used for command line compiling mode
        self.low_level_code_file = "none".encode("utf-8")
        # dllname is the name of the binary dll obtained after compilation, e.g. 7b9a611f7e.so
        self.dllname = self.gencode_file + ".so"
        # compile command string to obtain the dll, e.g. "g++ 7b9a611f7e.cpp -shared -fPIC -O3 -flto -o 7b9a611f7e.so"
        self.compile_command = f"{self.compiler} {' '.join(self.compile_options)} {self.gencode_file} -o {self.dllname}"
        # actual dll to be called 
        self.true_dllname = self.dllname
        # file to check for existence to detect compilation is needed
        self.file_to_check = self.dllname
        
    def compile_code(self):
        # method to generate the code and compile it
        # generate the code and save it in self.code, by calling get_code method from CpuReduc or GpuReduc classes :
        self.get_code()
        # write the code in the source file
        self.write_code()
        # call the compilation command
        os.system(self.compile_command)
        # retreive some parameters that will be saved into info_file.
        self.tagI = self.red_formula.tagI
        self.dim = self.red_formula.dim
        self.dimy = self.varloader.dimy
