import os

from keops.binders.LinkCompile import LinkCompile
from keops.config.config import cxx_compiler, cpp_flags


class Cpu_link_compile(LinkCompile):

    source_code_extension = "cpp"

    def __init__(self):
        LinkCompile.__init__(self)
        # these are used for command line compiling mode
        self.target_file = "none".encode("utf-8")
        # dllname is the name of the binary dll obtained after compilation, e.g. 7b9a611f7e.so
        self.dllname = self.gencode_file + ".so"
        # compile command string to obtain the dll, e.g. "g++ 7b9a611f7e.cpp -shared -fPIC -O3 -flto -o 7b9a611f7e.so"
        self.compile_command = (
            f"{cxx_compiler} {cpp_flags} {self.gencode_file} -o {self.dllname}"
        )
        # actual dll to be called
        self.true_dllname = self.dllname
        # file to check for existence to detect compilation is needed
        self.file_to_check = self.dllname

    def compile_code(self):
        # method to generate the code and compile it
        # generate the code and save it in self.code, by calling get_code method from CpuReduc class :
        self.get_code()
        # write the code in the source file
        self.write_code()
        # call the compilation command
        # os.system(self.compile_command)
        # retreive some parameters that will be saved into info_file.
        self.tagI = self.red_formula.tagI
        self.dim = self.red_formula.dim
